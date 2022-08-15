import os


def formatter(file_path):
    def eat(str, j, size):
        while j < size:
            if str[j] not in [' ', '\n']:
                break
            j += 1
        return j
    
    def str_equal(src, dst):
        if len(src) != len(dst):
            return False
        
        for p, q in zip(src, dst):
            if p != q:
                return False
        
        return True

    ignore_list = [':', '{', '}', ',', ' ', '\n']
    ret, indent, is_modify = '', '  ', False
    str = ''.join([l for l in open(file_path, 'r')])
    str = [e for e in str if e not in '\'\"']
    i, size, n_brack = 0, len(str), 0
    while i < size:
        e = str[i]

        if e not in ignore_list:
            ret += e

        if e == ':':
            ret += ': '
            i = eat(str, i + 1, size) - 1

        elif e == '{':
            n_brack += 1
            ret += '{\n' + indent * n_brack
            i = eat(str, i + 1, size) - 1

        elif e == '}':
            n_brack -= 1
            if ret[-1] not in ignore_list:
                ret += ','
            ret += '\n' + indent * n_brack + '}'
            if n_brack == 0:
                ret += '\n'
            i = eat(str, i + 1, size) - 1

        elif e == ',':
            i = eat(str, i + 1, size) - 1
            if str[i+1] == '}':
                ret += ','
            else:
                ret += ',\n' + indent * n_brack

        elif e == '\n' and n_brack == 0:
            ret += '\n'
            i = eat(str, i + 1, size) - 1

        i += 1
    
    if str[size-1] != '\n':
        ret += '\n'

    is_modify = False if str_equal(str, ret) else True

    if is_modify:
        with open(file_path, mode='w') as f:
            f.write(ret)

    return is_modify


def dfs_formatter(file_path):
    for e in os.listdir(file_path):
        if os.path.isdir(f'{file_path}/{e}'):
            dfs_formatter(f'{file_path}/{e}')
        else:
            if e.split('.')[-1] == 'yaml':
                if formatter(f'{file_path}/{e}'): 
                    print(f'[Formatter] {file_path}/{e}')


if __name__ == '__main__':
    yaml_base_path = './configs'
    dfs_formatter(yaml_base_path)
