"""Generate a tool-schema prompt with the pinned checkpoint encoder."""
import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('encoding',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
function={'description':'Lookup records','name':'lookup','parameters':{'properties':{'query':{'type':'string'}},'type':'object'}}
messages=[{'role':'system','content':'Use tools','tools':[{'type':'function','function':function}]},{'role':'user','content':'first'},{'role':'assistant','reasoning_content':'remember me','content':'answer'},{'role':'user','content':'again'}]
print(json.dumps({'function':function,'prompt':m.encode_messages(messages,thinking_mode='thinking',reasoning_effort=75)},ensure_ascii=False))
