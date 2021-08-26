import argparse
import subprocess

from expr import expr

def main(args):
    if args.plan:
        experiment = expr.__dict__[args.plan]()
        experiment.execute()
    elif args.list:
        exprlist = list(filter(lambda a: type(a) == type(object) and issubclass(a, expr.BaseExpr), expr.__dict__.values()))
        for experiment in exprlist:
            print(experiment.__name__)
    
    subprocess.run(["python3", "train.py", "-h"])


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='[Train Command Help]')
    args.add_argument('-p', '--plan', type=str,
                    help='Eperiment class name')
    args.add_argument('-l', '--list',  action='store_true',
                    help='View List of Eperiment class name')
    args = args.parse_args()
    
    main(args)