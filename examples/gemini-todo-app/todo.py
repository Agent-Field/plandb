import argparse
import db

def main():
    parser = argparse.ArgumentParser(description="Todo CLI App")
    subparsers = parser.add_subparsers(dest="command")

    # Add task
    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("title", help="Task title")

    # List tasks
    subparsers.add_parser("list")

    # Complete task
    complete_parser = subparsers.add_parser("complete")
    complete_parser.add_argument("id", type=int, help="Task ID")

    # Delete task
    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("id", type=int, help="Task ID")

    args = parser.parse_args()

    db.init_db()

    if args.command == "add":
        db.add_task(args.title)
        print(f"Added task: {args.title}")
    elif args.command == "list":
        tasks = db.list_tasks()
        for t in tasks:
            status = "[x]" if t[2] else "[ ]"
            print(f"{t[0]}. {status} {t[1]}")
    elif args.command == "complete":
        db.complete_task(args.id)
        print(f"Completed task {args.id}")
    elif args.command == "delete":
        db.delete_task(args.id)
        print(f"Deleted task {args.id}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
