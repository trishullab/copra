
import signal as sig
import hashlib
import contextlib
import re
import sys

from typing import (Optional, Tuple, TypeVar, Union, List, Pattern, Match)

from sexpdata import Symbol

T = TypeVar('T')


def unwrap(a: Optional[T]) -> T:
    assert a is not None
    return a


def split_by_char_outside_matching(openpat: str, closepat: str,
                                   splitpat: str, target: str) \
        -> Optional[Tuple[str, str]]:
    counter = 0
    curpos = 0
    with silent():
        openp = re.compile(openpat)
        closep = re.compile(closepat)
        splitp = re.compile(splitpat)

    def search_pat(pat: Pattern) -> Tuple[Optional[Match], int]:
        match = pat.search(target, curpos)
        return match, match.end() if match else len(target) + 1

    while curpos < len(target) + 1:
        _, nextopenpos = search_pat(openp)
        _, nextclosepos = search_pat(closep)
        nextsplitchar, nextsplitpos = search_pat(splitp)

        if nextopenpos < nextclosepos and nextopenpos < nextsplitpos:
            counter += 1
            assert nextopenpos > curpos
            curpos = nextopenpos
        elif nextclosepos < nextopenpos and \
                (nextclosepos < nextsplitpos or
                 (nextclosepos == nextsplitpos and counter > 0)):
            counter -= 1
            assert nextclosepos > curpos
            curpos = nextclosepos
        else:
            if counter <= 0:
                if nextsplitpos > len(target):
                    return None
                assert nextsplitchar
                return (target[:nextsplitchar.start()],
                        target[nextsplitchar.start():])
            else:
                assert nextsplitpos > curpos
                curpos = nextsplitpos
    return None


def eprint(*args, **kwargs):
    if "guard" not in kwargs or kwargs["guard"]:
        print(*args, file=sys.stderr,
              **{i: kwargs[i] for i in kwargs if i != 'guard'})
        sys.stderr.flush()


mybarfmt = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'


BLOCKSIZE = 65536


def hash_file(filename: str) -> str:
    hasher = hashlib.md5()
    with open(filename, 'rb') as f:
        buf = f.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(BLOCKSIZE)
    return hasher.hexdigest()


@contextlib.contextmanager
def sighandler_context(signal, f):
    old_handler = sig.signal(signal, f)
    yield
    sig.signal(signal, old_handler)


def progn(*args):
    return args[-1]


parsePat = re.compile("[() ]", flags=(re.ASCII | re.IGNORECASE))


def parseSexpOneLevel(sexp_str: str) -> Union[List[str], int, Symbol]:
    sexp_str = sexp_str.strip()
    if sexp_str[0] == '(':
        items = []
        cur_pos = 1
        item_start_pos = 1
        paren_level = 0
        while True:
            next_match = parsePat.search(sexp_str, cur_pos)
            if not next_match:
                break
            cur_pos = next_match.end()
            if sexp_str[cur_pos-1] == "(":
                paren_level += 1
            elif sexp_str[cur_pos-1] == ")":
                paren_level -= 1
                if paren_level == 0:
                    items.append(sexp_str[item_start_pos:cur_pos])
                    item_start_pos = cur_pos
            else:
                assert sexp_str[cur_pos-1] == " "
                if paren_level == 0:
                    items.append(sexp_str[item_start_pos:cur_pos])
                    item_start_pos = cur_pos
    elif re.fullmatch(r"\d+", sexp_str):
        return int(sexp_str)
    elif re.fullmatch(r"\w+", sexp_str):
        return Symbol(sexp_str)
    else:
        assert False, f"Couldn't parse {sexp_str}"
    return items


class DummyFile:
    def write(self, x): pass
    def flush(self): pass


@contextlib.contextmanager
def silent():
    save_stderr = sys.stderr
    save_stdout = sys.stdout
    sys.stderr = DummyFile()
    sys.stdout = DummyFile()
    yield
    sys.stderr = save_stderr
    sys.stdout = save_stdout
