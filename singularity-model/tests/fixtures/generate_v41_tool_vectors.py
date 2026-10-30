"""Generate tool completion vectors using the pinned checkpoint encoding.py."""
import importlib.util, json, sys
spec=importlib.util.spec_from_file_location('encoding',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cases=[
 {'name':'single','mode':'chat','reasoning':'','content':'Checking','calls':[{'name':'search','arguments':{'query':'café', 'limit':3}}]},
 {'name':'multi','mode':'thinking','reasoning':'Need data','content':'','calls':[{'name':'math::sum','arguments':{'items':[1,2], 'strict':True}},{'name':'ping','arguments':{}}]},
 {'name':'plain','mode':'chat','reasoning':'','content':'Done','calls':[]},
]
for case in cases:
    # Rust's serde_json map uses lexicographic key order by default.
    tc=[]
    for call in case['calls']:
        call['arguments']=dict(sorted(call['arguments'].items()))
        tc.append(m.tool_call_template.format(dsml_token=m.dsml_token,tool_call_tag_name=m.tool_call_tag_name,name=call['name'],arguments=m.encode_arguments_to_dsml(call)))
    calls=('\n\n'+m.tool_calls_template.format(dsml_token=m.dsml_token,tc_block_name=m.tool_calls_block_name,tool_calls='\n'.join(tc))) if tc else ''
    text=(case['reasoning']+'</think>' if case['mode']=='thinking' else '')+case['content']+calls+m.eos_token
    parsed=m.parse_message_from_completion_text(text,case['mode']); case['encoded_calls']=calls; case['completion']=text; case['parsed']=parsed
print(json.dumps(cases,ensure_ascii=False))
