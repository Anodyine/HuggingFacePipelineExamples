import threading


def count_operation():
    for i in range(30):
        print(threading.current_thread().name + ' - ' + str(i) + "\n")


# demo of sequential execution
# count_operation()
# count_operation()

# demo of parallel execution
t1 = threading.Thread(target=count_operation, name='First Thread')
t2 = threading.Thread(target=count_operation, name='Second Thread')

t1.start()
t2.start()
