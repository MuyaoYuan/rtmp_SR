import argparse
import template

parser = argparse.ArgumentParser(description='rtmp_SR')

parser.add_argument('-tp','--template', default='.',
                    help='the model you want to use')
parser.add_argument('-t','--task', default='.',
                    help='the task you want to do')

args = parser.parse_args()
template.set_template(args)