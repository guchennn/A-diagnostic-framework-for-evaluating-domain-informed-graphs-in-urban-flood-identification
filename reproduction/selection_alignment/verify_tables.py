"""Check the paired attribute contrasts in the supplied aggregate evidence."""
from pathlib import Path
import csv,json,math
root=Path(__file__).resolve().parent
def read(name):
    with (root/name).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
absolute={(r["city"],r["series"],r["variant"]):r for r in read("absolute_summary.csv")}
contrasts={(r["city"],r["series"],r["contrast"]):r for r in read("contrasts_summary.csv")}
count=0
for (city,series,contrast),r in contrasts.items():
    a,b=contrast.split(" - ")
    for field in ("mean","fold0","fold1","fold2","fold3","fold4"):
        want=float(absolute[city,series,a][field])-float(absolute[city,series,b][field])
        assert math.isclose(float(r[field]),want,abs_tol=1e-12), (city,series,contrast,field)
        count+=1
    assert math.isclose(float(r["mean"]),sum(float(r[f"fold{i}"]) for i in range(5))/5,abs_tol=1e-12)
for r in read("derived_summary.csv"):
    city,series,quantity=r["city"],r["series"],r["quantity"]
    if quantity.startswith("change from 18: "):
        contrast=quantity.removeprefix("change from 18: ")
        budget=series.rsplit(":",1)[1]
        a=contrasts[city,series,contrast]
        b=contrasts[city,f"attr:F18:{budget}",contrast]
        for field in ("mean","fold0","fold1","fold2","fold3","fold4"):
            assert math.isclose(float(r[field]),float(a[field])-float(b[field]),abs_tol=1e-12)
            count+=1
selections=read("independent_selections.csv")
fits=read("refit_metadata.csv")
assert len(selections)==520 and len(fits)==1560
assert sum(r["budget"]=="FULL" for r in fits)==360
assert len(contrasts)==24
print(json.dumps({"status":"PASS","paired_numeric_identities":count,
                  "conditions":24,"recorded_selections":520,"fit_metadata":1560,
                  "scope":"Aggregate identities and exported metadata; no new refitting."}))
