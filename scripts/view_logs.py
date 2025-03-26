import argparse
from reasoning.logs.activation_log import ActivationLogStore


def main():
    parser = argparse.ArgumentParser(description="View activation logs")

    parser.add_argument("--all", action="store_true", help="Show all logs")
    parser.add_argument("--config", type=str, help="Filter logs by SAE config name")
    parser.add_argument("--layer", type=int, help="Filter logs by layer index")
    parser.add_argument("--latest", action="store_true", help="Show latest log for given config")

    args = parser.parse_args()
    store = ActivationLogStore()

    if args.all:
        logs = store.all()
        print_logs(logs)

    elif args.latest and args.config:
        latest = store.latest_for(args.config)
        if latest:
            print_log(latest)
        else:
            print(f"No log found for config: {args.config}")

    elif args.config:
        logs = store.filter_by_config(args.config)
        print_logs(logs)

    elif args.layer is not None:
        logs = store.filter_by_layer(args.layer)
        print_logs(logs)

    else:
        parser.print_help()


def print_log(log):
    print(f"\n[✓] {log.created_at}")
    print(f"    Config: {log.sae_config}")
    print(f"    File:   {log.activations_file}")
    print(f"    Skipped: {log.skipped}")


def print_logs(logs):
    if not logs:
        print("No logs found.")
    for log in logs:
        print_log(log)


if __name__ == "__main__":
    main()
