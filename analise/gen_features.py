from subprocess import run, PIPE, STDOUT
from re import compile
from sys import argv
import os

def clean():
    current_dir = os.getcwd()
    files = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]
    for file in files:
        if "temp" in file:
            os.remove(file)

def call_savilerow(eprime, param):
    command = ["savilerow", eprime, param, "-chuffed"]
    process = run(command, stdout=PIPE, stderr=STDOUT, check=True, encoding="UTF-8")
    pattern = "Created output file (.*fzn)"
    prog = compile(pattern)
    for line in process.stdout.splitlines():
        match = prog.match(line)
        if match:
            return match.group(1)

def call_conjure(eprime, param):
    command = ["conjure", "translate-parameter", f"--eprime={eprime}", f"--essence-param={param}", f"--eprime-param=./temp.eprime-param"]
    process = run(command, stdout=PIPE, stderr=STDOUT, check=True, encoding="UTF-8")
    if process.stdout != "":
        raise Exception(process.stdout)

def call_fzn2feat(model_file):
    command = ["fzn2feat", model_file, "dict"]
    process = run(command, stdout=PIPE, stderr=STDOUT, check=True, encoding="UTF-8")
    return process.stdout

def main():
    if len(argv) < 2:
        print("error, please pass sthe necessary parameters. Use --help if needed")
        return
    if argv[1] == "--help" or argv[1] == "-h":
        print("python gen_feature eprime_file param_file")
        return
    if len(argv) < 3:
        print("error, please pass sthe necessary parameters. Use --help if needed")
        return

    eprime_file, param_file = argv[1], argv[2]
    call_conjure(eprime_file, param_file)
    generated_file = call_savilerow(eprime_file, "./temp.eprime-param")
    res = call_fzn2feat(generated_file)
    f = open(f"{eprime_file.split('/')[-1]}_{param_file.split('/')[-1]}.json", "w")
    f.write(res)
    f.close()
    clean()

if __name__ == "__main__":
    main()
