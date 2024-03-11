import threading


def count_operation():
    for i in range(100):
        print(i)


t1 = threading.Thread(target=count_operation, name='First Thread')
t2 = threading.Thread(target=count_operation, name='Second Thread')

t1.start()
t2.start()

t1.join()
t2.join()

print("Finished.")
