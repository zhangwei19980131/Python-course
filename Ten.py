command = ""
started = False
while True:  # 保证不管是输入的是大写小写都可以读入
    command = input("> ").lower()  # 简化写法
    if command == "start":
        if started:
            print("Car is already started!")
        else:
            print("Car started...")
    elif command == "stop":
        if not started:
            print("Car is already stopped!")
        else:
            started = False
            print("Car stopped.")
    elif command == "help":
        print("""
        start - to start the car
        stop - to stop the car
        quit - to quit
        """)
    elif command == "quit":
        break
    else:
        print("Sorry, I don't understand that...")