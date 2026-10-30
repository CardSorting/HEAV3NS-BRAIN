import importlib.util, json, sys
from tokenizers import Tokenizer
s=importlib.util.spec_from_file_location('reference',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
t=Tokenizer.from_file(sys.argv[2])
cases=[
 ('chat_simple','chat',75,[{'role':'system','content':'You are helpful.'},{'role':'user','content':'Hello!'}]),
 ('thinking_simple','thinking',75,[{'role':'system','content':'You are helpful.'},{'role':'user','content':'你好！'}]),
 ('multi_chat','chat',50,[{'role':'user','content':'One?'},{'role':'assistant','content':'First.'},{'role':'user','content':'Two?'}]),
 ('thinking_history','thinking',60,[{'role':'user','content':'First?'},{'role':'assistant','content':'Yes.','reasoning_content':'Thinking.'},{'role':'user','content':'Next?'}]),
 ('thinking_mid_system','thinking',100,[{'role':'user','content':'First?'},{'role':'assistant','content':'OK','reasoning_content':'Prior'},{'role':'system','content':'New instruction'}]),
 ('chat_system_only','chat',75,[{'role':'system','content':'Instruction'}]),
 ('thinking_assistant_last','thinking',75,[{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello','reasoning_content':'Simple'}])]
res=[]
for name,mode,effort,messages in cases:
 prompt=m.encode_messages(messages,thinking_mode=mode,reasoning_effort=effort)
 res.append({'name':name,'mode':mode,'effort':effort,'messages':messages,'prompt':prompt,'token_ids':t.encode(prompt,add_special_tokens=False).ids})
print(json.dumps(res,ensure_ascii=False,indent=2))
