from tokenizers import Tokenizer, Regex, normalizers
import hashlib,struct,sys
x=Tokenizer.from_file(sys.argv[1])
n=normalizers.Sequence([normalizers.NFKC(),normalizers.NFD(),normalizers.StripAccents(),normalizers.Lowercase(),normalizers.Replace(Regex(r'[ \t\r\n]+'),' '),normalizers.Replace(Regex(r'^ $'),'\ue000'),normalizers.Strip(),normalizers.Replace('\ue000',' ')])
keys={}; ids=[]
for i in range(x.get_vocab_size()):
    text=x.decode([i],skip_special_tokens=False)
    key=x.id_to_token(i) if '\ufffd' in text else (n.normalize_str(text) or text)
    if key not in keys: keys[key]=len(keys)
    ids.append(keys[key])
print('count',len(ids),'compressed',len(keys),'pad',ids[2]); print('sha256',hashlib.sha256(b''.join(struct.pack('<I',i) for i in ids)).hexdigest()); print('samples', [(i,ids[i]) for i in [0,1,2,3,10,100,1000,10000,129279]])
