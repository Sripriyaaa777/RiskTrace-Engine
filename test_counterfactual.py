from fastapi.testclient import TestClient

import main


def main_test() -> int:
    with TestClient(main.app) as client:
        valid = client.post("/counterfactual/KAFKA-23", json={"resolve_as": "Done"})
        if valid.status_code != 200:
            print(f"FAIL: expected 200 for valid counterfactual, got {valid.status_code}")
            print(valid.text)
            return 1

        impact = valid.json()["data"]["impact_summary"]
        if impact["nodes_improved"] <= 0 or impact["high_risk_reduction"] <= 0:
            print(f"FAIL: expected KAFKA-23 to improve downstream risk, got {impact}")
            return 1
        if impact["total_delay_days_saved"] < 0:
            print(f"FAIL: delay-days saved must not be negative, got {impact}")
            return 1

        missing = client.post("/counterfactual/NO-SUCH-ISSUE", json={"resolve_as": "Done"})
        if missing.status_code != 404:
            print(f"FAIL: expected 404 for missing issue, got {missing.status_code}")
            print(missing.text)
            return 1

    print("PASS: counterfactual endpoint handles valid and missing issue IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
