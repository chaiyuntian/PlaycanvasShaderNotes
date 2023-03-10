shader_chunk_root = 'engine\\src\\scene\\shader-lib\\chunks\\'

import os
import re


def filter_js_files(root_folder):
    js_files = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.js'):
                js_files.append(os.path.join(subdir, file))
    return js_files


def extract_shader_code(jsfile):
    # define the regular expression pattern to match the shader code
    pattern = re.compile(r'export default \/\* glsl \*\/`\n((.|\n)*)\n`;$')
    
    # load the js file as a string
    with open(jsfile, 'r') as f:
        js_code = f.read()
        # Extract the shader code using the regular expression
        match = re.search(pattern, js_code)
        print(match)
        if match != None:
            shader_code = match.group(1)
            return shader_code


def wrap_shader_code_to_markdown(shadercode):
    return "```glsl" + "\n" + shadercode + "\n" + "```"


def chunks_to_markdown(root_folder):
    jsfiles = filter_js_files(root_folder)
    with open("shader_chunks.md", "w") as f:
        for js in jsfiles:
            print(js)
            shader_code = extract_shader_code(js)
            
            if shader_code!=None and len(shader_code) > 0:
                f.write("\n" + js +  "\n")
                print(shader_code)
                f.write(wrap_shader_code_to_markdown(shader_code))
        f.close()
                
                
if __name__  == "__main__":
    chunks_to_markdown(shader_chunk_root)
