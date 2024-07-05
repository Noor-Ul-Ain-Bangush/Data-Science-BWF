
tasks = []
while True:
    print("\n<<----TO-DO LIST---->>")
    index = 1
    for task in tasks:
        if task["completed"]:
            status = "Completed"
        else:
            status = "Incomplete"

        print(f"{index}. {task['task']} | Due: {task['due_date']} | Status: {status}")
        index += 1

    print("\nOptions:")
    print("1.Add Task")
    print("2.Remove Task")
    print("3.Mark task as completed")
    print("4.exit")

    choice = input("\nEnter your choice: ")

    if choice == '1':
        task_name = input("Enter the task: ")
        due_date = input("Enter the due date: ")
        task = {
            "task" : task_name,
            "completed" : False,
            "due_date" : due_date,
        }
        tasks.append(task)
        print("Task added successfully!!!")

    elif choice == '2':
        task_num = int(input("Enter the task you want to remove: "))
        if 1 <= task_num <= len(tasks):
            tasks.pop(task_num-1)
            print("Task removed successfully!!!")
        else:
            print("Invalid")

    elif choice == '3':
        task_num = int(input("Enter the task number you want to mark as completed: "))
        if 1 <= task_num <= len(tasks):
            tasks[task_num-1]["completed"] = True
            print("Task marked as completed!!!")
        else:
            print("Invalid")

    elif choice == '4':
        print("Exit!!!")
        break

    else:
        print("Invalid....!!!")


