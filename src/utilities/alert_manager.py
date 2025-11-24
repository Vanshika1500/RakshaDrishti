import winsound

class AlertManager:
    def __init__(self):
        # You can set frequencies for different alerts
        self.low_freq = 1000  # Hz
        self.high_freq = 2000 # Hz
        self.duration = 300   # ms per beep

    def low_beep(self):
        winsound.Beep(self.low_freq, self.duration)

    def high_beep(self):
        winsound.Beep(self.high_freq, self.duration)
