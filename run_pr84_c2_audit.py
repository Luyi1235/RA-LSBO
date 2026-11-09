"""PR84 CLI: read-only audit of archived PR83 C2 evidence."""
import argparse,json
from pathlib import Path
from pr84_c2_audit import audit,RESULT_HEAD
def main():
 p=argparse.ArgumentParser(description="Read-only PR83 C2 archive audit")
 p.add_argument("--input",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 p.add_argument("--expected-result-head",default=RESULT_HEAD)
 a=p.parse_args()
 if a.expected_result_head!=RESULT_HEAD:p.error("PR83 result HEAD mismatch")
 r=audit(a.input,a.output)
 print(json.dumps({"status":r["status"],"full_files":r["full_files"],
  "warm_groups":r["warm_groups"],"cold_groups":r["cold_groups"],
  "all_six_arms_final_feasible":r["all_six_arms_final_feasible"],
  "integrity_flag":r["integrity_check_arms_all_final_feasible"],
  "integrity_flag_consistent":r["integrity_flag_consistent"],
  "new_math_calls":0,"new_gp_fits":0},sort_keys=True,indent=2))
if __name__=="__main__":main()
