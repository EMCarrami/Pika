from pika.utils.helpers import cli_parser
from pika.utils.run_utils import run_test

if __name__ == "__main__":
    config = cli_parser()
    run_test(config)
