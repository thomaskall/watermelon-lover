import serial
import io

ser = serial.Serial(
    port='/dev/cu.usbserial-1130',
    baudrate=9600,
    timeout=1
)
while(1):
    line = ser.readline()
    print(line)