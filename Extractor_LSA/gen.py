import argparse
import jsonlines

javascript_keywords = ["break", "case", "catch", "continue", "default", "delete", "do", "else", "finally", "for",
                       "function", "if", "in", "instanceof", "new", "return", "switch", "this", "throw", "try",
                       "typeof", "var", "void", "while", "with"]
java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
                 "continue", "default", "do", "double", "else", "enum", "extends", "false", "final", "finally", "float",
                 "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long", "native", "new",
                 "null", "package", "private", "protected", "public", "return", "short", "static", "strictfp", "super",
                 "switch", "synchronized", "this", "throw", "throws", "transient", "true", "try", "void", "volatile",
                 "while"]
python_keywords = ["and", "or", "not", "if", "elif", "else", "for", "while", "True", "False", "continue", "break",
                   "pass", "try", "except", "finally", "raise", "import", "from", "as", "def", "return", "class",
                   "lambda", "del", "global", "nonlocal", "in", "is", "None", "assert", "with", "yield", "async",
                   "await"]
php_keywords = ["abstract", "and", "array", "as", "break", "callable", "case", "catch", "class", "clone", "const",
                "continue", "declare", "default", "die", "do", "echo", "else", "elseif", "empty", "enddeclare", "eval",
                "exit", "endfor", "endforeach", "endif", "endswitch", "endwhile", "entends", "final", "finally", "fn",
                "for", "foreach", "function", "global", "goto", "if", "implements", "include", "instanceof",
                "insteadof", "interface", "isset", "list", "match", "namespace", "new", "or", "print", "private",
                "protected", "public", "readonly", "require", "return", "static", "switch", "throw", "trait", "try",
                "unset", "use", "var", "while", "xor", "yield"]
ruby_keywords = ["encoding", "line", "file", "begin", "end", "alias,", "and", "break", "case", "class", "def",
                 "defined", "do", "else", "elsif", "end", "ensure", "false", "for", "if", "in", "module", "next", "nil",
                 "not", "or", "redo", "rescue", "retry", "return", "self", "super", "then", "true", "undef", "unless",
                 "until", "when", "while", "yield"]
go_keywords = ['break', 'default', 'func', 'select', 'case', 'chan', 'interface', 'const', 'continue', 'defer', 'go',
               'map', 'struct', 'switch', 'if', 'else', 'goto', 'package', 'fallthrough', 'var', 'return', 'import',
               'type', 'range', 'for']


def add_args(parser):
    parser.add_argument("--file_path", type=str, default='')
    parser.add_argument("--ex_file_path", type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    file_path = args.file_path
    ex_file_path = args.ex_file_path

    with jsonlines.open(file_path, "r") as source, \
            open(ex_file_path, "r") as ex_file:
        writer = jsonlines.open(file_path, "w")
        result = []
        ex = ex_file.readlines()
        count = 0
        for obj in source:
            dic = {"idx": obj["idx"], "code_tokens": obj["code_tokens"], "docstring_tokens": obj["docstring_tokens"]
                , "cleaned_nl": obj["cleaned_nl"], "cleaned_codes": obj["cleaned_codes"],
                   "cleaned_seqs": obj["cleaned_seqs"]}
            ex_words_ = ex[count].split()[1:]
            ex_words = []
            for word in ex_words_:
                if word not in go_keywords:
                    ex_words.append(word)
            if len(ex_words) > 5:
                ex_words = ex_words[0:5]
            dic["extractive sum"] = ex_words
            statements = obj["cleaned_seqs"]
            my_pred_cleaned_seqs = [0] * len(statements)
            for k in range(0, len(statements)):
                stat_words = statements[k].split()
                for word in ex_words:
                    if word in stat_words:
                        my_pred_cleaned_seqs[k] = 1
                        break
            dic["my_pred_cleaned_seqs"] = my_pred_cleaned_seqs
            result.append(dic)
            count += 1
        for res in result:
            jsonlines.Writer.write(writer, res)
