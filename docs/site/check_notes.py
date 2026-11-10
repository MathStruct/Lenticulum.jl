import re,glob,os,collections,subprocess
tracked=subprocess.check_output(['git','-c','core.quotepath=off','ls-files','-co','--exclude-standard']).decode().split('\n')
mds=[f for f in tracked if f.endswith('.md')]
names={os.path.splitext(os.path.basename(n))[0] for n in mds}
notes=[f for f in mds if not f.startswith('docs/')]
broken=collections.defaultdict(set)
for n in notes:
    t=open(n).read()
    t=re.sub(r'```.*?```','',t,flags=re.S)
    for m in re.finditer(r'\[\[([^\]|#]+)',t):
        tgt=m.group(1).strip().split('/')[-1]
        if tgt not in names: broken[tgt].add(n)
print("BROKEN WIKILINKS (target not a tracked .md basename):",len(broken))
for k,v in sorted(broken.items()): print("  [[%s]] <- %s"%(k,sorted(v)[:3]))
roots=['','lib/LenticulumCore.jl/','lib/Mycelium.jl/','lib/MyceliumLearning.jl/','lib/Lenticulum.jl/','lib/VariationalDiffusion.jl/','lib/ImplicitLayers.jl/','lib/Adversarial.jl/','lib/AlgebraicFactors.jl/','docs/']
pat=re.compile(r'`((?:lib|src|test|bench|examples|docs|ext|vault)/[^`\s<>*]+?\.(?:jl|md|py|toml|sh|bib|yml|ts))`')
miss=collections.defaultdict(set)
ts=set(tracked)
for n in notes:
    t=open(n).read()
    for m in pat.finditer(t):
        p=m.group(1)
        if not any((r+p) in ts for r in roots) and not any(os.path.exists(r+p) for r in roots): miss[p].add(n)
print("MISSING FILE REFS (not found under repo root or any lib/*/):",len(miss))
for k,v in sorted(miss.items()): print("  %s <- %s"%(k,[os.path.basename(x) for x in sorted(v)][:3]))
# impl notes pairing
for d in glob.glob('lib/*/src'):
    jl={os.path.splitext(os.path.basename(f))[0] for f in glob.glob(d+'/*.jl')}
    md={os.path.splitext(os.path.basename(f))[0] for f in glob.glob(d+'/*.md')}
    print(d,"jl without md:",sorted(jl-md)," md without jl:",sorted(md-jl))
print("ext dirs md:", [ (d,len(glob.glob(d+'/*.jl')),len(glob.glob(d+'/*.md'))) for d in ['ext']+glob.glob('lib/*/ext')])
