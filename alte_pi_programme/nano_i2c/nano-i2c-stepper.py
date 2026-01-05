from gpiozero import LED
from time import sleep

# Wir missbrauchen SDA (Pin 3 / GPIO 2) und SCL (Pin 5 / GPIO 3) als LEDs
sda = LED(17) 
scl = LED(22)

print("Teste Pins... (Bitte Multimeter oder LED an Pin 3 & GND halten)")

while True:
    sda.on()
    scl.on()
    print("AN  (3.3V)")
    sleep(5)
    sda.off()
    scl.off()
    print("AUS (0V)")
    sleep(5)