import pandas as pd, re, json
S="sexist"; NS="non-sexist"; ID="ideological-inequality"; SD="stereotyping-dominance"
AUG=[
("Women are naturally better suited for nurturing roles than leadership positions.",S,ID),
("Men are just more logical than women, it's biology.",S,ID),
("A woman's place is really in the home, raising children.",S,ID),
("Women's sports just aren't as entertaining to watch as men's.",S,SD),
("She's a good player, but let's be honest, no woman could compete with the men's league.",S,SD),
("Female athletes get paid less because their sport simply doesn't generate the same revenue, it's not sexism.",S,ID),
("I respect women's basketball, but the athleticism just isn't comparable to men's.",S,SD),
("She's impressive for a female athlete.",S,SD),
("It's so nice to see women trying to play a man's sport.",S,SD),
("Wow, a girl who can actually play a real sport.",S,SD),
("She's pretty athletic for a woman.",S,SD),
("It's cute that she thinks she can compete at this level.",S,SD),
("Not bad for a girl, honestly.",S,SD),
("I'm surprised she's actually good, most women aren't into sports like this.",S,SD),
("She plays like one of the guys, which is a compliment.",S,SD),
("Women just aren't built for contact sports, it's a safety thing really.",S,ID),
("Men's sports will always be more competitive, it's just how it is.",S,ID),
("She's a great coach, pretty impressive for a woman in that role.",S,SD),
("Girls' sports are great for building confidence, even if the skill level is lower.",S,SD),
("It's refreshing to see a woman who isn't too emotional to handle pressure.",S,ID),
("She's tough, almost like a man out there.",S,SD),
("Women's leagues are fine, but they'll never draw the crowds men's sports do.",S,ID),
("I'll admit, she's better than I expected for a female athlete.",S,SD),
("It's nice that women have their own version of the sport to enjoy.",S,SD),
("She's a natural leader, surprisingly, for someone so soft-spoken as most women are.",S,ID),
("Women's sports deserve some attention, even if it's not the same level as men's.",S,ID),
("I didn't expect her to actually have game, most girls just play for fun.",S,SD),
("She's a solid athlete, especially considering she's a woman.",S,SD),
("Honestly impressive that a woman can keep up at this pace.",S,SD),
("Women's basketball is sweet to watch, like a lighter version of the real game.",S,SD),
("The team played really well this season and deserves more recognition.",NS,NS),
("That was an incredible comeback in the fourth quarter.",NS,NS),
("She's one of the best point guards in the league right now.",NS,NS),
("This rivalry has been great for the sport overall.",NS,NS),
("The coaching staff made some smart adjustments at halftime.",NS,NS),
("Attendance numbers have been climbing steadily this season.",NS,NS),
("That injury is going to be tough for the team to recover from.",NS,NS),
("The new stadium has really improved the fan experience.",NS,NS),
("Ticket prices have gone up a lot compared to last year.",NS,NS),
("The playoffs this year have been unusually competitive.",NS,NS)]
# held-out test set: the user's 20 examples + the "girl who can actually play a real sport" example
TEST=["Women are naturally better suited for nurturing roles than leadership positions.","Men are just more logical than women, it’s biology.","A woman’s place is really in the home, raising children.","Women’s sports just aren’t as entertaining to watch as men’s.","She’s a good player, but let’s be honest, no woman could compete with the men’s league.","Female athletes get paid less because their sport simply doesn’t generate the same revenue, it’s not sexism.","I respect women’s basketball, but the athleticism just isn’t comparable to men’s.","She’s impressive for a female athlete.","It’s so nice to see women trying to play a man’s sport.","The team played really well this season and deserves more recognition.","Women are too stupid to ever lead anything, biology doesn’t lie.","Men are obviously smarter than women, anyone who disagrees is delusional.","Get back in the kitchen where you belong, you idiot.","Women’s sports are a joke, nobody wants to watch that garbage.","She’s trash compared to any real athlete, women can’t compete with men, period.","Female athletes don’t deserve a cent, their sport is worthless and boring.","Women’s basketball is a pathetic excuse for a sport, it’s an embarrassment.","She’s actually decent for a woman, shocking honestly.","Wow, a girl who can actually play a real sport, who would’ve thought","This team is garbage and doesn’t deserve any recognition.","Wow, a girl who can actually play a real sport."]
norm=lambda t: re.sub(r"[^a-z0-9 ]","",t.lower().replace("’","'")).strip()
tn={norm(t) for t in TEST}
excluded,kept=[],[]
for t,l,c in AUG:
    n=norm(t)
    exact=n in tn
    near=any(n!=x and (n.startswith(x[:len(n)]) or x.startswith(n)) and min(len(n),len(x))>25 for x in tn)  # near-duplicate (one is a prefix of the other)
    (excluded if exact or near else kept).append((t,l,c,"exact" if exact else "near-dup"))
print("EXCLUDED (held out as test-only):",len(excluded))
for t,*_,why in [(e[0],e[3]) for e in excluded]: print(f"  [{why}] {t}")
d=pd.read_csv('EXIST2021-2025_datasets/2021 EXIST/training/EXIST2021_training.tsv',sep='\t')
new=pd.DataFrame([{"test_case":"AUG","id":90001+i,"source":"handwritten","language":"en","text":t,"task1":l,"task2":c} for i,(t,l,c,_) in enumerate(kept)])
out=pd.concat([d,new],ignore_index=True); out.to_csv('augmented/EXIST2021_training_augmented.tsv',sep='\t',index=False)
print(f"\noriginal {len(d)} + kept {len(new)} = {len(out)}  (sexist added {sum(k[1]==S for k in kept)}, non-sexist added {sum(k[1]==NS for k in kept)})")
print("kept for training:"); [print("  ",k[0]) for k in kept]
json.dump(TEST,open('augmented/heldout_test_sentences.json','w'),ensure_ascii=False,indent=1)
