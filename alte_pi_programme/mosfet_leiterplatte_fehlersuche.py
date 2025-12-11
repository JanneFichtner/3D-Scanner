import lgpio, sys, termios, tty

p1=18
p2=27
h=lgpio.gpiochip_open(0)

lgpio.gpio_claim_output(h,p1) 

lgpio.gpio_claim_output(h,p2)

fd=sys.stdin.fileno()
old=termios.tcgetattr(fd)
tty.setraw(fd)
try:
    while True:
        k=sys.stdin.read(1)
        if k=="\r": lgpio.gpio_write(h,p1,1); lgpio.gpio_write(h,p2,1)
        if k==" ":  lgpio.gpio_write(h,p1,0); lgpio.gpio_write(h,p2,0)
        if k=="x": break
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    lgpio.gpiochip_close(h)




