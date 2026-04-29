import re, os
with open(r'D:\College\Sem VI\Minor Project\flowlevel\FlowLevel_Review2_Presentation.html', encoding='utf-8') as f:
    content = f.read()
slides = len(re.findall(r'id="s\d+"', content))
figures = re.findall(r'src="figures/(.*?)"', content)
print(f"Slides found: {slides}")
print(f"Figures referenced: {figures}")
fig_dir = r'D:\College\Sem VI\Minor Project\flowlevel\figures'
all_ok = True
for fig in figures:
    exists = os.path.exists(os.path.join(fig_dir, fig))
    print(f"  {fig}: {'OK' if exists else 'MISSING'}")
    if not exists: all_ok = False
print(f"File size: {len(content)} bytes")
print(f"All figures present: {all_ok}")
