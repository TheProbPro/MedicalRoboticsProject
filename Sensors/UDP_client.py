import ctypes
import socket
import struct
import time
from multiprocessing import Process, Array, Event

import numpy as np


class UDPSensor:
    def __init__(self):
        # Device IP and port
        self.HOST_IP = "192.168.1.1"
        self.PORT = 49152  # Device listening on port 49151

        # Variables for running parallel process
        self.data = Array(ctypes.c_uint32, range(9))
        self.stopEvent = Event()

        self.start_command, self.response_format = self._request_selector(command_num=1)
        self.p = Process(target=self.acquire_data, args=[self.data, self.stopEvent, self.HOST_IP, self.PORT])

    def start(self, unbias_data=False):
        """Method to start sensor to output data with defined settings and create process to collect data

        :return:
        """
        # if sensor output should be unbiased to set current values to zero
        if unbias_data:
            self.set_bias(bias=255)
        self._send_command(command=self._request_selector(command_num=4)[0])
        # send start command
        print("starting")
        self.p.start()

    def stop(self):
        """Method to stop sensor from outputting data and kill data collecting process

        :return: nothing
        """
        print("stopping")
        self.stopEvent.set()
        self.p.join()

    def get(self):
        """Method for obtaining data from continuous acquisition

        :return:
        """
        convert = np.array([1, 1, 1, 1e4, 1e4, 1e4, 1e5, 1e5, 1e5], dtype=np.int32)
        result = np.array(self.data[:], dtype=np.uint32)
        return np.where(result > np.iinfo(np.int32).max, result - np.iinfo(np.uint32).max, result).astype(np.int32)/convert

    def set_bias(self, bias=0):
        """Method  for setting custom bias given in parameter

        :param bias: [0-255 decimal] (0-reset values, 255-current values as bias)
        :return:
        """
        if bias in range(256):
            command = struct.pack("!HHI", 0x1234, 0x0042, bias)
            self._send_command(command=command)
        else:
            print(f"Wrong bias ({bias}). Select 0-255")

    def send_custom_command(self):
        """Method for manual command selector and sending

        :return:
        """
        command, response_format = self._request_selector()
        self._send_command(command=command)

    def _send_command(self, command):
        """Method for sending given command in binary format

        :param command: command in binary format according to datasheet
        :return: nothing
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect((self.HOST_IP, self.PORT))
            sock.sendall(command)
            print("Request sent to device.")
            sock.close()
        except Exception as e:
            print(e)

    def _request_selector(self, command_num=None):
        """Request command selector invoke user input selector if command_num=None

            :param command_num: number of command in available command database
            :return: nothing
            """
        # check if the command number was already chosen
        command_selected = False if command_num is None else True

        # define available commands and their parameters
        sample_count = 0  # number of samples as a threshold to the amount of data to be sent (0 means send until stop command)
        sw_bias = 255  # [0-255 decimal] (0-reset values, 255-current values as bias)
        int_filtering = 4  # [0-6 decimal] cut-off frequencies (0=none, 1=500Hz, 2=150Hz, 3=50Hz, 4=15Hz(default), 5=5Hz, 6=1.5Hz)
        read_speed = 2  # [255-1 (0-stops read-out)] period in ms - formula: 1000 Hz / new_value = new_frequency (10ms default)
        header = 0x1234  # is fixed
        available_commands = {"Stop sending the output": [0x0000, [0], '!HHI', None],
                              "Start sending the output ": [0x0002, [sample_count], '!HHI', '!IIIIIIIII'],
                              "Set software bias": [0x0042, [sw_bias], '!HHI', None],
                              "Set internal filtering": [0x0081, [int_filtering], '!HHI', None],
                              "Set read-out speed": [0x0082, [read_speed], '!HHI', None], }
        pairing = [*available_commands.keys()]
        command = None
        data = None

        # Cycle through menu and user selector until valid command is chosen.
        while not command_selected:

            # Show available commands
            print("Available commands:")
            for i, c in enumerate(available_commands.keys()):
                print(f"{i}) - {c}")
            print("\n")

            # Get user input for commands
            try:
                user_input = int(input("Select command type: "))
            except Exception as e:
                print(e)
                continue

            # validate user input
            if (user_input >= 0) and (user_input < len(pairing)):
                command_num = user_input
                command_selected = True
            else:
                print(f"Command under selected number \"{user_input}\" not available.")
                continue

        # create command
        command, data, request_format, response_format = available_commands[pairing[command_num]]
        request_message = struct.pack(request_format, header, command, *data)
        return request_message, response_format

    def acquire_data(self, data, stop_event, host_ip, port):
        """ Method for parallel data acquisition in separate process

        :param port: host port
        :param host_ip: host ip address
        :param data: shared variable to share state between processes
        :param stop_event: stop event to stop data collection when stopEvent.is_set() by .set()
        :return: nothing, data is returned through shared variable data
        """
        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)

        try:
            # Connect to the device
            sock.connect((host_ip, port))
            print(f"Connected to {host_ip,}:{port}")

            # send start command
            sock.sendall(self.start_command)
            # Run infinite loop to collect data
            count = 0
            while not stop_event.is_set():
                # Wait for the response (36 bytes)
                response = sock.recv(36)
                # print(response)

                if len(response) == 36:
                    # Parse the response based on the provided structure
                    HS_sequence, FT_sequence, status, fx, fy, fz, tx, ty, tz = struct.unpack('!IIIIIIIII', response)
                    data[:] = [HS_sequence, FT_sequence, status, fx, fy, fz, tx, ty, tz]
                    # print(count)
                    # count+=1
                    # print(f"Response received:")
                    # print(f"  HS_sequence: 0x{HS_sequence:04X}")
                    # print(f"  FT_sequence: 0x{FT_sequence:04X}")
                    # print(f"  Status: 0x{status:04X}")
                    # print(f"  Fx: {fx / 10000}")
                    # print(f"  Fy: {fy / 10000}")
                    # print(f"  Fz: {fz / 10000}")
                    # print(f"  Tx: {tx / 100000}")
                    # print(f"  Ty: {ty / 100000}")
                    # print(f"  Tz: {tz / 100000}")
                else:
                    print("Received an invalid response size.")
                    print(response)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        except Exception as e:
            print(e)
        finally:
            # Close the socket connection
            sock.close()
            print("Connection closed.")


if __name__ == '__main__':
    # test()
    sensor = UDPSensor()
    # sensor.send_custom_command()
    sensor.start(unbias_data=True)
    start = time.time()
    convert = np.array([1,1,1,1e4,1e4,1e4,1e5,1e5,1e5], dtype=np.int32)
    for i in range(100):

        # print(np.array(sensor.get(), dtype=np.uint32).view(np.int32))
        # arr = []
        # for s in sensor.get():
        #     if s > np.iinfo(np.int32).max:
        #         arr.append(s -np.iinfo(np.uint32).max)
        #     else:
        #         arr.append(s)
        print(sensor.get()[3:6])
        # print(arr)
        # print([if i>0x7FFFFFFF: i-0x100000000 else: i ])
        time.sleep(0.1)
    print(f"end: {time.time()-start} s")
    sensor.stop()