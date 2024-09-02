import os
import pty
import subprocess
import json
import select
import time
import logging

class ProcessInterface:
    buffer_size = 1024
    def __init__(self, command, cwd, logger: logging.Logger = None, log_level=logging.INFO, reboot_every_n_commands = 50):
        """
        Note: This class is not thread-safe. It is intended to be used in a single-threaded environment.
        """
        assert reboot_every_n_commands > 0
        self.reboot_every_n_commands = reboot_every_n_commands
        self.command = command
        self.cwd = cwd
        self._last_reboot = 0
        self._start()
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.setLevel(log_level)
    
    def _start(self):
        master, slave = pty.openpty()
        self.process = subprocess.Popen(
            self.command.split(),
            cwd=self.cwd,
            stdin=slave,
            stdout=slave,
            stderr=subprocess.STDOUT,
            text=True
        )
        os.close(slave)
        self.master = master
        self.buffer = ''  # Buffer to accumulate data from stdout
        self.sent_commands = ''  # Buffer to track sent commands
        pass

    def is_rebooted(self):
        return self._last_reboot == 0

    def send_command(self, command_dict):
        # Check if the process is still running
        if not self.process_is_running():
            raise Exception("Process is not running.")
        json_command = json.dumps(command_dict, ensure_ascii=False) + '\n\n'
        normalized_command = json_command.replace('\r\n', '\n')  # Normalize newlines
        os.write(self.master, normalized_command.encode('utf-8'))  # Send command to process
        self.logger.debug(f"Sent: {normalized_command}")
        self.sent_commands += normalized_command  # Keep track of normalized sent commands
        time.sleep(0.3) # Wait for the process to process the command
    
    def process_is_running(self):
        return self.process.poll() is None

    def read_response(self, timeout=10):
        try:
            response = self._read_response(timeout)
            return response
        except:
            self.buffer = ''
            self.sent_commands = ''
            raise

    def _read_response(self, timeout=10):
        end_time = time.time() + timeout
        wait_time = 0.2
        response_read = False
        input_was_read = False
        output_was_read = False
        while time.time() < end_time and not response_read:
            readable, _, _ = select.select([self.master], [], [], wait_time)
            # print("readable: ", readable)
            while len(readable) > 0 and not response_read:
                try:
                    data = os.read(self.master, ProcessInterface.buffer_size).decode('utf-8', 'ignore')  # Read data from stdout
                    if input_was_read:
                        output_was_read = True
                except OSError as e:
                    # Ignore the os error
                    if e.errno == 5:
                        response_read = True
                        break
                    else:
                        raise
                # print("data: ", data)
                normalized_data = data.replace('\r\n', '\n')  # Normalize received data
                self.buffer += normalized_data
                # Clean buffer by removing echoed commands before parsing
                if self.sent_commands != '' and self.buffer.startswith(self.sent_commands):
                    self.buffer = self.buffer[len(self.sent_commands):]  # Remove echoed commands
                    self.sent_commands = ''  # Clear sent commands buffer after removing
                    input_was_read = True
                if input_was_read and self.buffer.find('{') != -1 and self.buffer.rfind('}') != -1:
                    output_was_read = True
                readable, _, _ = select.select([self.master], [], [], wait_time)
                response_read = len(readable) == 0 and output_was_read
                if time.time() >= end_time:
                    response_read = False
                    break
        self.buffer = self.buffer.strip()  # Remove leading/trailing whitespace
        if not response_read or len(self.buffer) == 0 or self.buffer.find('{') == -1:
            self.logger.debug("Could not read response. Waiting for more data.")
            raise TimeoutError(f"Could not read response. Buffer: {self.buffer}")
        self.buffer = self.buffer[self.buffer.index('{'):]  # Remove any leading garbage
        try:
            # Attempt to parse the clean buffer as JSON
            response = json.loads(self.buffer)
            self.logger.debug(f"Received: {response}")
            self.buffer = ''  # Clear buffer after successful parse
            self._last_reboot += 1
            if self._last_reboot >= self.reboot_every_n_commands:
                self.close()
                self._start()
                self._last_reboot = 0
            return response
        except json.JSONDecodeError as e:
            self.logger.debug("Failed to parse JSON. Waiting for more data.")
            self.logger.debug(f"Buffer: {self.buffer}")
            raise TimeoutError(f"Failed to parse JSON. Buffer: {self.buffer}", e)

    def close(self):
        os.close(self.master)
        self.process.terminate()
        self.process.wait()


# Process interface test
if __name__ == "__main__":
    #.lake/bin/repl
    repl_path = "./src/tools/repl/.lake/build/bin/repl"
    lean4_proj_path = "./data/test/lean4_proj"
    file_path = "./data/test/lean4_proj/Lean4Proj/Basic.lean"
    # lean4_proj_path = './src/data/test/Mathlib'
    # file_path = './src/data/test/Mathlib/.lake/packages/mathlib/Mathlib/Data/Nat/Bits.lean'
    file_path = os.path.abspath(file_path)
    abs_repl_path = os.path.abspath(repl_path)
    interface = ProcessInterface(f"lake env {abs_repl_path}", lean4_proj_path, log_level=logging.DEBUG, reboot_every_n_commands=1)
    try:
        interface.send_command({"path": file_path, "allTactics": True})
        response = interface.read_response(1000)
        print(response)
        print("====================================")
        interface.send_command({"path": file_path, "allTactics": True})
        response = interface.read_response(1000)
        print(response)
    finally:
        interface.close()