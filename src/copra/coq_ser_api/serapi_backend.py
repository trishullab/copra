
import threading
import re
import subprocess
import queue
import signal
import functools
import os
import os.path
from typing import Optional, List, Any, TYPE_CHECKING, cast

from sexpdata import Symbol, loads, dumps, ExpectClosingBracket
from pampy import match, _, TAIL

from .coq_backend import (CoqBackend, CoqAnomaly, CompletedError,
                          AckError, CoqExn, BadResponse,
                          CoqTimeoutError, ParseError,
                          UnrecognizedError, CoqException,
                          NoSuchGoalError)
from .contexts import ProofContext, Obligation, SexpObligation
from .coq_util import raise_, parsePPSubgoal, setup_opam_env, get_module_from_filename
from .util import (eprint, parseSexpOneLevel, unwrap, progn)
if TYPE_CHECKING:
    from sexpdata import Sexp

class CoqSeraPyInstance(CoqBackend, threading.Thread):

    def __init__(self, coq_command: List[str],
                 timeout: int = 30, set_env: bool = True) -> None:

        if set_env:
            setup_opam_env()
        self.__sema = threading.Semaphore(value=0)
        threading.Thread.__init__(self, daemon=True)
        self.version_string = subprocess.run(["sertop", "--version"], stdout=subprocess.PIPE,
                                             text=True, check=True).stdout
        if self.version_string.strip() == "":
            self.version_string = "8.10.0"
            print(f"Using dev version of sertop, setting version string to {self.version_string}")
        assert self.coq_minor_version() >= 10, \
            "Versions of Coq before 8.10 are not supported! "\
            f"Currently installed coq is {self.version_string}"
        if self.coq_minor_version() <= 12:
            self.all_goals_regex = all_goals_regex_10
        else:
            self.all_goals_regex = all_goals_regex_13

        # Open a process to coq, with streams for communicating with
        # it.
        self.root_dir = "."
        self._proc = subprocess.Popen(coq_command,
            # " ".join(coq_command) if isinstance(coq_command, list) else coq_command,
            # shell=True,
            cwd=".",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        self._fout = self._proc.stdout
        self._fin = self._proc.stdin

        # Initialize some state that we'll use to keep track of the
        # coq state. This way we don't have to do expensive queries to
        # the other process to answer simple questions.
        self.proof_context: Optional[ProofContext] = None
        self.cur_state = 0
        self.timeout = timeout

        # Set up the message queue, which we'll populate with the
        # messages from serapi.
        self.message_queue = queue.Queue()  # type: queue.Queue[str]
        # Verbosity is zero until set otherwise
        self.verbosity = 0
        # Set the "extra quiet" flag (don't print on failures) to false
        self.quiet = False
        # The messages printed to the *response* buffer by the command
        self.feedbacks: List[Any] = []
        # Start the message queue thread
        self.start()
        # Go through the messages and throw away the initial feedback.
        self._discard_feedback()

    def addStmt_noupdate(self, stmt: str, timeout:Optional[int] = None) -> None:
        assert self.proof_context
        if stmt.strip() == "":
            return
        if timeout:
            old_timeout = self.timeout
            self.timeout = timeout
        self._flush_queue()
        assert self.message_queue.empty(), self.messages
        stmt = stmt.replace("\\", "\\\\")
        stmt = stmt.replace("\"", "\\\"")
        self._send_acked(f"(Add () \"{stmt}\")\n")
        # Get the response, which indicates what state we put
        # serapi in.
        self._update_state()
        self._get_completed()
        if timeout:
            self.timeout = old_timeout
        assert self.message_queue.empty()
    def updateState(self) -> None:
        # Execute the statement.
        self._send_acked("(Exec {})\n".format(self.cur_state))
        # Finally, get the result of the command
        self.feedbacks = self._get_feedbacks()
        # Get a new proof context, if it exists
        self._get_proof_context(update_nonfg_goals=True)

    def addStmt(self, stmt: str, timeout:Optional[int] = None,
                force_update_nonfg_goals: bool = False) -> None:
        if stmt.strip() == "":
            return
        if timeout:
            old_timeout = self.timeout
            self.timeout = timeout
        try:
            self._flush_queue()
            assert self.message_queue.empty(), self.messages
            stmt = stmt.replace("\\", "\\\\")
            stmt = stmt.replace("\"", "\\\"")
            self._send_acked(f"(Add () \"{stmt}\")\n")
            # Get the response, which indicates what state we put
            # serapi in.
            self._update_state()
            self._get_completed()
            assert self.message_queue.empty()
            # Track goal opening/closing
            is_goal_open = re.match(r"\s*(?:\d+\s*:)?\s*[{]\s*", stmt)
            is_goal_close = re.match(r"\s*[}]\s*", stmt)
            is_unshelve = re.match(r"\s*Unshelve\s*\.\s*", stmt)
            is_bullet = re.match(r"\s*[-+*]+", stmt)

            # Execute the statement.
            self._send_acked(f"(Exec {self.cur_state})\n")
            # Finally, get the result of the command
            self.feedbacks = self._get_feedbacks()
            # Get a new proof context, if it exists
            if is_goal_open:
                self._get_enter_goal_context()
            elif is_goal_close or is_unshelve or is_bullet:
                self._get_proof_context(update_nonfg_goals=True)
            else:
                self._get_proof_context(update_nonfg_goals=force_update_nonfg_goals)
        except (CoqExn, BadResponse, AckError,
                CompletedError, CoqTimeoutError) as e:
            self._handle_exception(e, stmt)
        finally:
            if timeout:
                self.timeout = old_timeout
    def cancelLastStmt_noupdate(self, cancelled: str) -> None:
        self._flush_queue()
        assert self.message_queue.empty(), self.messages
        # Run the cancel
        self._send_acked("(Cancel ({}))".format(self.cur_state))
        # Get the response from cancelling
        self.cur_state = self._get_cancelled()

    def cancelLastStmt(self, cancelled: str, force_update_nonfg_goals: bool = False) -> None:
        is_goal_open = re.match(r"\s*(?:\d+\s*:)?\s*[{]\s*", cancelled)
        is_goal_close = re.match(r"\s*[}]\s*", cancelled)
        is_unshelve = re.match(r"\s*Unshelve\s*\.\s*", cancelled)
        is_bullet = re.match(r"\s*[-+*]+", cancelled)
        self.__cancel()
        # Get a new proof context, if it exists
        self._get_proof_context(update_nonfg_goals=
                      bool(is_goal_open or is_goal_close or
                           is_unshelve or is_bullet or
                           force_update_nonfg_goals))

    def getProofContext(self) -> Optional[ProofContext]:
        return self.proof_context

    def close(self) -> None:
        assert self._proc.stdout
        self._proc.terminate()
        self._proc.kill()
        self.__sema.release()
    def isInProof(self) -> bool:
        return self.proof_context is not None
    def queryVernac(self, vernac: str) -> List[str]:
        self._send_acked(f"(Query () (Vernac \"{vernac}\"))")
        next_msg = self._get_message()
        while isProgressMessage(next_msg):
            next_msg = self._get_message()
        feedbacks = []
        while self._isFeedbackMessage(next_msg):
            feedbacks.append(next_msg)
            next_msg = self._get_message()
        # Only for older coq versions?
        # This case is here because older versions of Coq/Serapi send an
        # extra message after the feedbacks when the response is bad.
        if self.coq_minor_version() <= 12:
            def handle_bad_response(_: Any):
                # All (Answer {int} (ObjList ...)) messages are followed by
                # an (Answer {int} Completed) message.
                # in order to keep future commands from failing, we need to
                # consume the Completed message.
                self._get_completed()
                raise BadResponse(next_msg)
            
            match(normalizeMessage(next_msg),
                  ["Answer", int, ["ObjList", []]], lambda state: None,
                  _, handle_bad_response)

        self._get_completed()
        return ["\n".join(searchStrsInMsg(msg)) for msg in feedbacks]
    def interrupt(self) -> None:
        self._proc.send_signal(signal.SIGINT)
        self._flush_queue()
    def enterDirectory(self, new_dir: str) -> None:
        self.root_dir = os.path.join(self.root_dir, new_dir)
        try:
            with open(self.root_dir + "/_CoqProject", 'r') as includesfile:
                includes_string = includesfile.read()
        except FileNotFoundError:
            try:
                with open(self.root_dir + "/Make", 'r') as includesfile:
                    includes_string = includesfile.read()
            except FileNotFoundError:
                eprint(f"Didn't find _CoqProject or Make for {self.root_dir}",
                       guard=self.verbosity)
                includes_string = ""

        self.addStmt(f"Cd \"{new_dir}\".")

        q_pattern = r"-Q\s*(\S+)\s+(\S+)\s*"
        r_pattern = r"-R\s*(\S+)\s+(\S+)\s*"
        i_pattern = r"-I\s*(\S+)\s*"
        for includematch in re.finditer(rf"({q_pattern})|({r_pattern})|({i_pattern})",
                                        includes_string):
            q_match = re.fullmatch(r"-Q\s*(\S*)\s*(\S*)\s*", includematch.group(0))
            if q_match:
                if q_match.group(2) == "\"\"":
                    self.addStmt(
                        f"Add LoadPath \"{q_match.group(1)}\".")
                else:
                    self.addStmt(
                        f"Add LoadPath \"{q_match.group(1)}\" as {q_match.group(2)}.")
                continue
            r_match = re.match(r"-R\s*(\S*)\s*(\S*)\s*", includematch.group(0))
            if r_match:
                self.addStmt(
                    f"Add Rec LoadPath \"{r_match.group(1)}\" as {r_match.group(2)}.")
                continue
            i_match = re.match(r"-I\s*(\S*)", includematch.group(0))
            if i_match:
                self.addStmt(
                    f"Add ML Path \"{i_match.group(1)}\".")
                continue
    def setFilename(self, filename: str) -> None:
        module_name = get_module_from_filename(filename)
        self.addStmt(f"Module {module_name}.")
    def resetCommandState(self) -> None:
        self.addStmt("Reset Initial.")
        self.addStmt("Optimize Heap.", timeout=15)
        self.enterDirectory(".")

    def coq_minor_version(self) -> int:
        version_match = re.fullmatch(r"\d+\.(\d+).*", self.version_string,
                                     flags=re.DOTALL)
        assert version_match, f"Version {self.version_string} doesn't match regex"
        return int(version_match.group(1))
    @property
    def messages(self):
        return [dumps(msg) for msg in list(self.message_queue.queue)]
    @property
    def feedback_string(self):
        if len(self.feedbacks) < 4:
            return ""
        string_lists = [searchStrsInMsg(f) for f in self.feedbacks]
        nonempty_string_lists = [l for l in string_lists if len(l) > 0]
        return "\n".join([slist[0] for slist in nonempty_string_lists])
    def getSexpProofContext(self) -> List[SexpObligation]:
        assert self.proof_context, "Can only call get_all_sexp_goals when you're in a proof!"
        text_response = self._ask_text("(Query () Goals)")
        context_match = re.fullmatch(
            r"\(Answer\s+\d+\s*\(ObjList\s*(.*)\)\)\n",
            text_response)
        if not context_match:
            if "Stack overflow" in text_response:
                raise CoqAnomaly(f"\"{text_response}\"")
            else:
                raise BadResponse(f"\"{text_response}\"")
        context_str = context_match.group(1)
        assert context_str != "()"
        goals_match = self.all_goals_regex.match(context_str)
        if not goals_match:
            raise BadResponse(context_str)
        fg_goals_str, bg_goals_str, \
            shelved_goals_str, given_up_goals_str = \
            goals_match.groups()
        fg_goal_strs = cast(List[str], parseSexpOneLevel(fg_goals_str))
        bg_goal_strs = [uuulevel for ulevel in cast(List[str],
                                                    parseSexpOneLevel(bg_goals_str))
                        for uulevel in cast(List[str], parseSexpOneLevel(ulevel))
                        for uuulevel in cast(List[str], parseSexpOneLevel(uulevel))]
        if len(fg_goal_strs) > 0 or len(bg_goal_strs) > 0:
            goals: List[SexpObligation] = []
            for goal_str in fg_goal_strs + bg_goal_strs:
                loaded = loads(goal_str)
                goals.append(SexpObligation([['CoqConstr', ty[2]] for ty in loaded[2][1]],
                                            ['CoqConstr', loaded[1][1]]))
            return goals
        else:
            return []
    def backToState(self, state_num: int) -> None:
        self.addStmt(f"BackTo {state_num}.")
    def backToState_noupdate(self, state_num: int) -> None:
        self.addStmt_noupdate(f"BackTo {state_num}.")
    def _isFeedbackMessage(self, msg: str) -> bool:
        # if self.coq_minor_version() > 12:
        return isFeedbackMessage(msg)
        # return isFeedbackMessageOld(msg)
    def _flush_queue(self) -> None:
        while not self.message_queue.empty():
            self._get_message()
    def _discard_feedback(self) -> None:
        try:
            feedback_message = self._get_message()
            while feedback_message[1][3][1] != Symbol("Processed"):
                feedback_message = self._get_message()
        except CoqTimeoutError:
            pass
        except CoqAnomaly as e:
            if e.msg != "Timing Out":
                raise
    def _get_message(self, complete=False) -> Any:
        msg_text = self._get_message_text(complete=complete)
        assert msg_text != "None", msg_text
        if msg_text[0] != "(":
            eprint(f"Skipping non-sexp output {msg_text}",
                   guard=self.verbosity>=3)
            return self._get_message(complete=complete)
        try:
            return loads(msg_text, nil=None)
        except ExpectClosingBracket as exc:
            eprint(
                f"Tried to load a message but it's ill formed! \"{msg_text}\"",
                guard=self.verbosity)
            raise CoqAnomaly("") from exc
        except AssertionError as exc:
            eprint(f"Assertion error while parsing s-expr {msg_text}")
            raise CoqAnomaly("") from exc
    def _get_message_text(self, complete=False) -> Any:
        try:
            msg = self.message_queue.get(timeout=self.timeout)
            if complete:
                self._get_completed()
            assert msg is not None
            return msg
        except queue.Empty as exc3:
            eprint("Command timed out! Interrupting", guard=self.verbosity)
            self._proc.send_signal(signal.SIGINT)
            num_breaks = 1
            try:
                interrupt_response = \
                    loads(self.message_queue.get(timeout=self.timeout))
            except queue.Empty:
                self._proc.send_signal(signal.SIGINT)
                num_breaks += 1
                try:
                    interrupt_response = \
                        loads(self.message_queue.get(timeout=self.timeout))
                except queue.Empty as exc:
                    raise CoqAnomaly("Timing Out") from exc

            got_answer_after_interrupt = match(
                normalizeMessage(interrupt_response),
                ["Answer", int, ["CoqExn", TAIL]],
                lambda *args: False,
                ["Answer", TAIL],
                lambda *args: True,
                _, lambda *args: False)
            if got_answer_after_interrupt:
                self._get_completed()
                for _i in range(num_breaks):
                    try:
                        after_interrupt_msg = loads(self.message_queue.get(
                            timeout=self.timeout))
                    except queue.Empty as exc:
                        raise CoqAnomaly("Timing Out") from exc
                    assert isBreakMessage(after_interrupt_msg), \
                        after_interrupt_msg
                assert self.message_queue.empty(), self.messages
                return dumps(interrupt_response)
            for _i in range(num_breaks):
                try:
                    after_interrupt_msg = loads(self.message_queue.get(
                        timeout=self.timeout))
                except queue.Empty as exc2:
                    raise CoqAnomaly("Timing Out") from exc2
            self._get_completed()
            assert self.message_queue.empty(), self.messages
            raise CoqTimeoutError("") from exc3
    # Get the next message from the message queue, and make sure it's
    # a Completed.
    def _get_completed(self) -> None:
        completed = self._get_message()
        match(normalizeMessage(completed),
              ["Answer", int, "Completed"], lambda state: None,
              _, lambda msg: raise_(CompletedError(completed)))
    # Send some text to serapi, and flush the stream to make sure they
    # get it. NOT FOR EXTERNAL USE
    def _send_flush(self, cmd: str):
        assert self._fin
        eprint("SENT: " + re.sub("\n+", "", cmd), guard=self.verbosity >= 4)
        try:
            self._fin.write(cmd.encode('utf-8'))
            self._fin.flush()
        except BrokenPipeError as exc:
            raise CoqAnomaly("Coq process unexpectedly quit. Possibly running "
                             "out of memory due to too many threads?") from exc
    def _send_acked(self, cmd: str):
        self._send_flush(cmd)
        self._get_ack()
    def _get_ack(self) -> None:
        ack = self._get_message()
        match(normalizeMessage(ack),
              ["Answer", _, "Ack"], lambda state: None,
              ["Feedback", TAIL], lambda rest: self._get_ack(),
              _, lambda msg: raise_(AckError(dumps(ack))))
    def _update_state(self) -> None:
        self.cur_state = self._get_next_state()
    def _get_next_state(self) -> int:
        msg = self._get_message()
        while match(normalizeMessage(msg),
                    ["Feedback", TAIL], lambda tail: True,
                    ["Answer", int, "Completed"], lambda sidx: True,
                    _, lambda x: False):
            msg = self._get_message()

        return match(normalizeMessage(msg),
                     ["Answer", int, list],
                     lambda state_num, contents:
                     match(contents,
                           ["CoqExn", TAIL],
                           lambda rest:
                           raise_(CoqAnomaly("Overflowed"))
                           if "Stack overflow" in "\n".join(searchStrsInMsg(rest))
                           else raise_(CoqExn("\n".join(searchStrsInMsg(rest)))),
                           ["Added", int, TAIL],
                           lambda state_num, tail: state_num),
                     _, lambda x: raise_(BadResponse(msg)))
    def _get_feedbacks(self) -> List['Sexp']:
        unparsed_feedbacks: List[str] = []
        unparsed_next_message = self._get_message_text()
        while unparsed_next_message.startswith("(Feedback"):
            unparsed_feedbacks.append(unparsed_next_message)
            unparsed_next_message = self._get_message_text()
        fin = unparsed_next_message
        if re.match(r"\(Answer\s+\d+\s*\(CoqExn", fin):
            message = "\n".join(searchStrsInMsg(loads(unparsed_feedbacks[-1], nil=None)))
            if "Stack overflow" in message:
                raise CoqAnomaly("Overflowed")
            raise CoqExn(message)

        return [loads(feedback_text, nil=None) for feedback_text in unparsed_feedbacks]
    def _get_enter_goal_context(self) -> None:
        assert self.proof_context
        self.proof_context = ProofContext([self.proof_context.fg_goals[0]],
                                          self.proof_context.fg_goals[1:] +
                                          self.proof_context.bg_goals,
                                          self.proof_context.shelved_goals,
                                          self.proof_context.given_up_goals)
    def _get_proof_context(self, update_nonfg_goals: bool = True) -> None:
        # Try to do this the right way, fall back to the
        # wrong way if we run into this bug:
        # https://github.com/ejgallego/coq-serapi/issues/150
        try:
            text_response = self._ask_text("(Query () Goals)")
            context_match = re.fullmatch(
                r"\(Answer\s+\d+\s*\(ObjList\s*(.*)\)\)",
                text_response.strip())
            if not context_match:
                if "Stack overflow" in text_response:
                    raise CoqAnomaly(f"\"{text_response}\"")
                raise BadResponse(f"\"{text_response}\"")
            context_str = context_match.group(1)
            if context_str == "()":
                self.proof_context = None
            else:
                goals_match = self.all_goals_regex.match(context_str)
                if not goals_match:
                    raise BadResponse(context_str)
                fg_goals_str, bg_goals_str, \
                    shelved_goals_str, given_up_goals_str = \
                    goals_match.groups()
                if update_nonfg_goals or self.proof_context is None:
                    unparsed_levels = cast(List[str],
                                           parseSexpOneLevel(bg_goals_str))
                    parsed2 = [uuulevel
                               for ulevel in unparsed_levels
                               for uulevel in cast(List[str],
                                                   parseSexpOneLevel(ulevel))
                               for uuulevel in
                               cast(List[str], parseSexpOneLevel(uulevel))]
                    bg_goals = [self._parseSexpGoalStr(bg_goal_str)
                                for bg_goal_str in parsed2]
                    self.proof_context = ProofContext(
                        [self._parseSexpGoalStr(goal)
                         for goal in cast(List[str],
                                          parseSexpOneLevel(fg_goals_str))],
                        bg_goals,
                        [self._parseSexpGoalStr(shelved_goal)
                         for shelved_goal in
                         cast(List[str],
                              parseSexpOneLevel(shelved_goals_str))],
                        [self._parseSexpGoalStr(given_up_goal)
                         for given_up_goal in
                         cast(List[str],
                              parseSexpOneLevel(given_up_goals_str))])
                else:
                    self.proof_context = ProofContext(
                        [self._parseSexpGoalStr(goal)
                         for goal in cast(List[str],
                                          parseSexpOneLevel(fg_goals_str))],
                        unwrap(self.proof_context).bg_goals,
                        [self._parseSexpGoalStr(shelved_goal)
                         for shelved_goal in
                         cast(List[str],
                              parseSexpOneLevel(shelved_goals_str))],
                        unwrap(self.proof_context).given_up_goals)
        except CoqExn:
            self._send_acked("(Query ((pp ((pp_format PpStr)))) Goals)")

            msg = self._get_message()
            proof_context_msg = match(
                normalizeMessage(msg),
                ["Answer", int, ["CoqExn", TAIL]],
                lambda statenum, rest:
                raise_(CoqAnomaly("Stack overflow")) if
                "Stack overflow." in searchStrsInMsg(rest) else
                raise_(CoqExn("\n".join(searchStrsInMsg(rest)))),
                ["Answer", int, list],
                lambda statenum, contents: contents,
                _, lambda *args:
                raise_(UnrecognizedError(dumps(msg))))
            self._get_completed()
            if len(proof_context_msg) == 0:
                self.proof_context = None
            else:
                newcontext = self._extract_proof_context(proof_context_msg[1])
                if newcontext == "none":
                    self.proof_context = ProofContext([], [], [], [])
                else:
                    self.proof_context = \
                        ProofContext(
                            [parsePPSubgoal(substr) for substr
                             in re.split(r"\n\n|(?=\snone)", newcontext)
                             if substr.strip()],
                            [], [], [])
    def _handle_exception(self, e: CoqException, stmt: str):
        eprint(f"Problem running statement: {stmt}\n",
               guard=self.verbosity >= 2)
        match(e,
              CoqTimeoutError,
              lambda *args: progn(self.cancel_failed(),  # type: ignore
                                  raise_(CoqTimeoutError(
                                      f"Statment \"{stmt}\" timed out."))),
              _, lambda e: None)
        coqexn_msg = match(normalizeMessage(e.msg),
                           ['Answer', int, ['CoqExn', TAIL]],
                           lambda sentence_num, rest:
                           "\n".join(searchStrsInMsg(rest)),
                           str, lambda s: s,
                           [str], lambda s: s,
                           _, None)
        if coqexn_msg:
            eprint(coqexn_msg, guard=self.verbosity >= 2)
            if ("Stream\\.Error" in coqexn_msg
                    or "Syntax error" in coqexn_msg
                    or "Syntax Error" in coqexn_msg):
                self._get_completed()
                raise ParseError(f"Couldn't parse command {stmt}")
            if "CLexer.Error" in coqexn_msg:
                self._get_completed()
                raise ParseError(f"Couldn't parse command {stmt}")
            if "NoSuchGoals" in coqexn_msg:
                self._get_completed()
                self.cancel_failed()
                raise NoSuchGoalError("")
            if "Invalid_argument" in coqexn_msg:
                if "index out of bounds" in coqexn_msg and "Anomaly" in coqexn_msg:
                    self._get_completed()
                    self.cancel_failed()
                raise ParseError(f"Invalid argument in {stmt}")
            if "Not_found" in coqexn_msg:
                self._get_completed()
                self.cancel_failed()
                raise e
            if "Overflowed" in coqexn_msg or "Stack overflow" in coqexn_msg:
                self._get_completed()
                raise CoqAnomaly("Overflowed")
            if "Anomaly" in coqexn_msg:
                self._get_completed()
                raise CoqAnomaly(coqexn_msg)
            if "Unable to unify" in coqexn_msg:
                self._get_completed()
                self.cancel_failed()
                raise CoqExn(coqexn_msg)
            if re.match(r".*The identifier (.*) is reserved\..*",
                          coqexn_msg):
                self._get_completed()
                raise CoqExn(coqexn_msg)
            self._get_completed()
            self.cancel_failed()
            raise CoqExn(coqexn_msg)
        match(normalizeMessage(e.msg),
              ['Stream\\.Error', str],
              lambda *args: progn(self._get_completed(), # type: ignore
                                  raise_(ParseError(
                                      f"Couldn't parse command {stmt}"))),

              ['CErrors\\.UserError', _],
              lambda inner: progn(self._get_completed(), # type: ignore
                                  self.cancel_failed(),  # type: ignore
                                  raise_(e)),
              ['ExplainErr\\.EvaluatedError', TAIL],
              lambda inner: progn(self._get_completed(), # type: ignore
                                  self.cancel_failed(),  # type: ignore
                                  raise_(e)),
              _, lambda *args: progn(raise_(UnrecognizedError(args))))
    def _ask(self, cmd: str, complete: bool = True):
        return loads(self._ask_text(cmd, complete))

    def _ask_text(self, cmd: str, complete: bool = True):
        assert self.message_queue.empty(), self.messages
        self._send_acked(cmd)
        msg = self._get_message_text(complete)
        return msg
    def _ppStrToTermStr(self, pp_str: str) -> str:
        answer = self._ask(
            f"(Print ((pp ((pp_format PpStr)))) (CoqPp {pp_str}))")
        return match(normalizeMessage(answer),
                     ["Answer", int, ["ObjList", [["CoqString", _]]]],
                     lambda statenum, s: str(s),
                     ["Answer", int, ["CoqExn", TAIL]],
                     lambda statenum, msg:
                     raise_(CoqExn("\n".join(searchStrsInMsg(msg)))))

    def _ppToTermStr(self, pp) -> str:
        return self._ppStrToTermStr(dumps(pp))

    @functools.lru_cache(maxsize=128)
    def _sexpStrToTermStr(self, sexp_str: str) -> str:
        try:
            answer = self._ask(
                f"(Print ((pp ((pp_format PpStr)))) (CoqConstr {sexp_str}))")
            return match(normalizeMessage(answer),
                         ["Answer", int, ["ObjList", [["CoqString", _]]]],
                         lambda statenum, s: str(s),
                         ["Answer", int, ["CoqExn", TAIL]],
                         lambda statenum, msg:
                         raise_(CoqExn("\n".join(searchStrsInMsg(msg)))))
        except CoqExn as e:
            eprint("Coq exception when trying to convert to string:\n"
                   f"{sexp_str}", guard=self.verbosity >= 1)
            eprint(e, guard=self.verbosity >= 2)
            raise

    def _sexpToTermStr(self, sexp) -> str:
        return self._sexpStrToTermStr(dumps(sexp))

    def _parseSexpHypStr(self, sexp_str: str) -> str:
        var_sexps_str, _mid_str, term_sexp_str = \
            cast(List[str], parseSexpOneLevel(sexp_str))

        def get_id(var_pair_str: str) -> str:
            id_possibly_quoted = unwrap(
                id_regex.match(var_pair_str)).group(1)
            if id_possibly_quoted[0] == "\"" and \
               id_possibly_quoted[-1] == "\"":
                return id_possibly_quoted[1:-1]
            return id_possibly_quoted
        ids_str = ",".join([get_id(var_pair_str) for
                            var_pair_str in
                            cast(List[str], parseSexpOneLevel(var_sexps_str))])
        term_str = self._sexpStrToTermStr(term_sexp_str)
        return f"{ids_str} : {term_str}"

    def _parseSexpHyp(self, sexp) -> str:
        var_sexps, _, term_sexp = sexp
        ids_str = ",".join([dumps(var_sexp[1]) for var_sexp in var_sexps])
        term_str = self._sexpToTermStr(term_sexp)
        return f"{ids_str} : {term_str}"

    def _parseSexpGoalStr(self, sexp_str: str) -> Obligation:
        goal_match = goal_regex.fullmatch(sexp_str)
        assert goal_match, sexp_str + "didn't match"
        _goal_num_str, goal_term_str, hyps_list_str = \
            goal_match.group(1, 2, 3)
        goal_str = self._sexpStrToTermStr(goal_term_str).replace(r"\.", ".")
        hyps = [self._parseSexpHypStr(hyp_str) for hyp_str in
                cast(List[str], parseSexpOneLevel(hyps_list_str))]
        return Obligation(hyps, goal_str)

    def _parseSexpGoal(self, sexp) -> Obligation:
        _goal_num, goal_term, hyps_list = \
            match(normalizeMessage(sexp),
                  [["name", int], ["ty", _], ["hyp", list]],
                  lambda *args: args)
        goal_str = self._sexpToTermStr(goal_term)
        hyps = [self._parseSexpHyp(hyp_sexp) for hyp_sexp in hyps_list]
        return Obligation(hyps, goal_str)

    def _parseBgGoal(self, sexp) -> Obligation:
        return match(normalizeMessage(sexp),
                     [[], [_]],
                     self._parseSexpGoal)
    def _extract_proof_context(self, raw_proof_context: 'Sexp') -> str:
        assert isinstance(raw_proof_context, list), raw_proof_context
        assert len(raw_proof_context) > 0, raw_proof_context
        assert isinstance(raw_proof_context[0], list), raw_proof_context
        return cast(List[List[str]], raw_proof_context)[0][1]
    def cancel_failed(self) -> None:
        self.__cancel()
    def __cancel(self) -> None:
        self._flush_queue()
        assert self.message_queue.empty(), self.messages
        # Run the cancel
        self._send_acked(f"(Cancel ({self.cur_state}))")
        # Get the response from cancelling
        self.cur_state = self._get_cancelled()
    def _get_cancelled(self) -> int:
        try:
            feedback = self._get_message()

            new_statenum = \
                match(normalizeMessage(feedback),
                      ["Answer", int, ["CoqExn", TAIL]],
                      lambda docnum, rest:
                      raise_(CoqAnomaly("Overflowed"))
                      if "Stack overflow" in "\n".join(searchStrsInMsg(rest))
                      else raise_(CoqExn(feedback)),
                      ["Feedback", [['doc_id', int], ['span_id', int], TAIL]],
                      lambda docnum, statenum, *rest: statenum,
                      _, lambda *args: raise_(BadResponse(feedback)))

            cancelled_answer = self._get_message()

            match(normalizeMessage(cancelled_answer),
                  ["Answer", int, ["Canceled", list]],
                  lambda _, statenums: min(statenums),
                  ["Answer", int, ["CoqExn", TAIL]],
                  lambda statenum, rest:
                  raise_(CoqAnomaly("\n".join(searchStrsInMsg(rest))))
                  if "Anomaly" in "\n".join(searchStrsInMsg(rest)) else
                  raise_(CoqExn("\n".join(searchStrsInMsg(rest)))),
                  _, lambda *args: raise_(BadResponse(cancelled_answer)))
        finally:
            self._get_completed()

        return new_statenum
    def run(self) -> None:
        assert self._fout
        while not self.__sema.acquire(False):
            try:
                line = self._fout.readline().decode('utf-8')
            except ValueError:
                continue
            if line.strip() == '':
                break
            self.message_queue.put(line)
            eprint(f"RECEIVED: {line}", guard=self.verbosity >= 4)

def isFeedbackMessageOld(msg: 'Sexp') -> bool:
    return match(normalizeMessage(msg),
                 ["Feedback", [["doc_id", int], ["span_id", int],
                               ["route", int],
                               ["contents", ["Message", "Notice",
                                             [], TAIL]]]],
                 lambda *args: True,
                 _, lambda *args: False)
def isFeedbackMessage(msg: 'Sexp') -> bool:
    return match(normalizeMessage(msg, depth=6),
                 ["Feedback", [["doc_id", int], ["span_id", int],
                               ["route", int],
                               ["contents", ["Message", ["level", "Notice"],
                                             ["loc", []], TAIL]]]],
                 lambda *args: True,
                 _, lambda *args: False)

def isProgressMessage(msg: 'Sexp') -> bool:
    return match(normalizeMessage(msg),
                 ["Feedback", [["doc_id", int], ["span_id", int],
                               ["route", int],
                               ["contents", ["ProcessingIn", str]]]],
                 lambda *args: True,
                 ["Feedback", [["doc_id", int], ["span_id", int],
                               ["route", int],
                               ["contents", "Processed"]]],
                 lambda *args: True,
                 _,
                 lambda *args: False)

def isBreakMessage(msg: 'Sexp') -> bool:
    return match(normalizeMessage(msg),
                 "Sys.Break", lambda *args: True,
                 "Sys\\.Break", lambda *args: True,
                 _, lambda *args: False)

def normalizeMessage(sexp, depth: int = 5):
    if depth <= 0:
        return sexp
    if isinstance(sexp, list):
        return [normalizeMessage(item, depth=depth-1) for item in sexp]
    if isinstance(sexp, Symbol):
        return dumps(sexp)
    return sexp

def searchStrsInMsg(sexp, fuel: int = 30) -> List[str]:
    if isinstance(sexp, list) and len(sexp) > 0 and fuel > 0:
        if sexp[0] == "str" or sexp[0] == Symbol("str"):
            assert len(sexp) == 2 and (isinstance(sexp[1], str) or isinstance(sexp[1], Symbol)), sexp
            return [str(sexp[1])]
        return [substr
                for substrs in [searchStrsInMsg(sublst, fuel - 1)
                                for sublst in sexp]
                for substr in substrs]
    return []


goal_regex = re.compile(r"\(\(info\s*\(\(evar\s*\(Ser_Evar\s*(\d+)\)\)"
                        r"\(name\s*\((?:\(Id\"?\s*[\w']+\"?\))*\)\)\)\)"
                        r"\(ty\s*(.*)\)\s*\(hyp\s*(.*)\)\)")

all_goals_regex_10 = re.compile(r"\(\(CoqGoal\s*"
                                r"\(\(goals\s*(.*)\)"
                                r"\(stack\s*(.*)\)"
                                r"\(shelf\s*(.*)\)"
                                r"\(given_up\s*(.*)\)"
                                r"\(bullet\s*.*\)\)\)\)")

all_goals_regex_13 = re.compile(r"\(\(CoqGoal\s*"
                                r"\(\(goals\s*(.*)\)"
                                r"\(stack\s*(.*)\)"
                                r"\(bullet\s*.*\)"
                                r"\(shelf\s*(.*)\)"
                                r"\(given_up\s*(.*)\)\)\)\)")

id_regex = re.compile(r"\(Id\s*(.*)\)")
