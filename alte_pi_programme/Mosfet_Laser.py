
import lgpio
from time import sleep, perf_counter
import numpy as np

laser_pin = 18
laser_pin_01= 27
# GPIO-Chip öffnen (meistens gpiochip0)
h = lgpio.gpiochip_open(0)

# Pin als Ausgang claimen
lgpio.gpio_claim_output(h, laser_pin)

def schnellerwerden():
    for i in range(9,18, 2):
        print("T: ", 2 * ( 1 / i))
        for j in range (1, 10):
            lgpio.gpio_write(h, laser_pin, 1)
            sleep(1 / i)
            lgpio.gpio_write(h, laser_pin, 0)
            
            lgpio.gpio_write(h, laser_pin_01, 1)
            sleep(1 / i)
            lgpio.gpio_write(h, laser_pin_01, 0)
def gleichzeitig():
    lgpio.gpio_write(h, laser_pin, 0)
    lgpio.gpio_write(h, laser_pin_01, 0)
    sleep(1)
    lgpio.gpio_write(h, laser_pin, 1)
    sleep(1)
    lgpio.gpio_write(h, laser_pin_01, 1)
    sleep(1)
    lgpio.gpio_write(h, laser_pin, 0)
    lgpio.gpio_write(h, laser_pin_01, 0)
    lgpio.gpio_write(h, laser_pin, 0)
    lgpio.gpio_write(h, laser_pin_01, 0)
gleichzeitig()
schnellerwerden()



        

                                                                                        
# Chip wieder schließen
lgpio.gpiochip_close(h)
  