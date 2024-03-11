from threading import Thread


class Counter(Thread):

    def __init__(self, name):
        Thread.__init__(self)
        self.name = name

    def run(self):
        for i in range(100):
            print("%s thread is running: %s\n" % (self.name, str(i)))


t1 = Counter("First")
t2 = Counter("Second")

t1.start()
t2.start()
