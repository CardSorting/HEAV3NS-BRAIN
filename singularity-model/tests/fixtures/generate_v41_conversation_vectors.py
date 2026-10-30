"""Generate tool-conversation prompts with pinned checkpoint encoding.py."""
import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('encoding',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
call=lambda i,n,a: {'id':i,'type':'function','function':{'name':n,'arguments':json.dumps(a,ensure_ascii=False)}}
cases=[
 {'name':'results-out-of-order','mode':'thinking','messages':[{'role':'user','content':'Find data'}, {'role':'assistant','content':'Working','reasoning_content':'Need tools','tool_calls':[call('a','search',{'query':'alpha'}),call('b','math::add',{'x':2})]}, {'role':'tool','tool_call_id':'b','content':'2'}, {'role':'tool','tool_call_id':'a','content':'found'}]},
 {'name':'result-plus-user','mode':'chat','messages':[{'role':'user','content':'go'}, {'role':'assistant','content':'','tool_calls':[call('x','fetch',{})]}, {'role':'tool','tool_call_id':'x','content':'done'}, {'role':'user','content':'continue'}, {'role':'assistant','content':'ok'}]},
 {'name':'prior-reasoning','mode':'thinking','messages':[{'role':'user','content':'first'}, {'role':'assistant','content':'old','reasoning_content':'hidden'}, {'role':'user','content':'next'}]},
]
for case in cases: case['prompt']=m.encode_messages(case['messages'],thinking_mode=case['mode'],reasoning_effort=75)
print(json.dumps(cases,ensure_ascii=False))
