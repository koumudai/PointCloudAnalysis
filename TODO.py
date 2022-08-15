import os

ignore_list = ['nothing.py', 'nothing.yaml', 'TODO.md', 'TODO.py', '__pycache__']
ignore_list.extend(['data', 'datasets', 'models', 'utils', 'losses'])

def vis(path, l):
    for e in os.listdir(path):
        if e in ignore_list:
            continue

        if os.path.isdir(f'{path}/{e}'):
            print('│   ' * (l-1) + '├──' + e + '/')
            vis(f'{path}/{e}', l+1)
        else:
            print('│   ' * (l-1) + '├──' + e)
            

if __name__ == '__main__':
    path = 'F:/Workspace/Github/Papers/Point_Cloud/Code'
    vis(path, 1)

    