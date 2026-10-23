#!/usr/bin/env python3
"""Opt-in, sequential Jev calls with durable responses and a per-run budget.

No automatic retries. No API key is stored, printed, or accepted as a command
argument. The key is read, in order, from the TYPESAFE_API_KEY environment
variable, the project's .env file (TYPESAFE_API_KEY=...), or a hidden prompt.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

from .study import PINNED_MODEL, PRICE_PER_MILLION, RESERVED_TOKENS_PER_CALL, digest, object_hash, read_jsonl, validate_answer

API_URL="https://api.typesafe.ai/v1/systemone"
from .paths import ENV_FILE  # <project root>/.env
KEY_NAME="TYPESAFE_API_KEY"


def env_file_value(name=KEY_NAME,path=ENV_FILE):
    """Return NAME's value from a simple KEY=VALUE .env file, or None.

    Supports blank lines, # comments, an optional 'export ' prefix and
    single/double quotes. The value is never printed or logged.
    """
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line=line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line=line[7:].lstrip()
        field,value=line.split("=",1)
        if field.strip()!=name:
            continue
        value=value.strip()
        if len(value)>=2 and value[0]==value[-1] and value[0] in "\"'":
            value=value[1:-1]
        elif " #" in value:
            value=value.split(" #",1)[0].rstrip()
        return value or None
    return None


def resolve_key():
    key=os.environ.get(KEY_NAME) or env_file_value()
    if key:
        return key
    return getpass.getpass("Jev API key (hidden; not saved): ")


def append(path,record):
    with path.open("a",encoding="utf-8") as handle:
        handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
        handle.flush()
        os.fsync(handle.fileno())


def budget_used(events):
    starts={e["attempt_id"]:e for e in events if e["event"]=="started"}
    finished={e["attempt_id"]:e for e in events if e["event"]=="finished"}
    total=known=uncertain=0.
    for ident,event in starts.items():
        completion=finished.get(ident,{})
        actual=completion.get("actual_cost_usd")
        if actual is None:
            uncertain+=event["reserved_usd"]
        else:
            known+=actual
    total=known+uncertain
    return {"accounted_usd":total,"known_list_price_usd":known,"uncertain_reserved_usd":uncertain,
            "attempts":len(starts)}


def fetch(body,key,timeout):
    request=urllib.request.Request(API_URL,data=json.dumps(body).encode(),headers={
        "Authorization":"Bearer "+key,"Content-Type":"application/json","Accept":"application/json"},method="POST")
    started=time.perf_counter()
    with urllib.request.urlopen(request,timeout=timeout) as response:
        payload=response.read()
    parsed=json.loads(payload)
    # Full response time includes HTTP round trip, body read and JSON decoding.
    return parsed,(time.perf_counter()-started)*1000


def run(folder,budget=1.,execute=False,max_requests=None,pause=.05,timeout=30.,retry_failed=False,fetcher=fetch):
    if not 0 < budget <= 5:
        raise ValueError("Choose a positive per-run budget no larger than the stated $5 credit allocation")
    manifest=json.loads((folder/"data-manifest.json").read_text())
    if digest(folder/"requests.jsonl")!=manifest["requests_sha256"]:
        raise ValueError("Prepared request hash changed")
    requests=read_jsonl(folder/"requests.jsonl")
    if any(r["request"]["model"]!=PINNED_MODEL for r in requests):
        raise ValueError("Requests must pin the documented model version")
    reserve=RESERVED_TOKENS_PER_CALL*PRICE_PER_MILLION/1_000_000
    if not execute:
        print(json.dumps({"mode":"dry_run","requests":len(requests),"model":PINNED_MODEL,
                          "budget_usd":budget,"maximum_reserved_input_tokens_per_call":RESERVED_TOKENS_PER_CALL,
                          "reservation_usd_per_call":reserve,"model_calls":0,
                          "next_step":"Add --execute to opt in to billable calls. Enter your key at the hidden local prompt."},indent=2))
        return {"status":"dry_run","successes":0,"expected":len(requests)}
    lock=folder/".jev.lock"
    try:
        handle=lock.open("x")
    except FileExistsError:
        raise ValueError("Runner lock exists. Ensure no runner is active before removing .jev.lock") from None
    try:
        with handle:
            handle.write(str(os.getpid()))
        meta_path=folder/"jev-config.json"
        config={"model":PINNED_MODEL,"request_file_sha256":digest(folder/"requests.jsonl"),
                "input_usd_per_million":PRICE_PER_MILLION,"reserve_tokens_per_call":RESERVED_TOKENS_PER_CALL,
                "endpoint":API_URL,"concurrency":1,"automatic_retries":0}
        if meta_path.exists() and json.loads(meta_path.read_text())!=config:
            raise ValueError("Existing Jev run uses a different frozen configuration")
        meta_path.write_text(json.dumps(config,indent=2)+"\n")
        journal=folder/"jev-attempts.jsonl"
        events=read_jsonl(journal) if journal.exists() else []
        responses_path=folder/"jev-responses.jsonl"
        saved=read_jsonl(responses_path) if responses_path.exists() else []
        success={r["id"]:r for r in saved if r["status"]=="ok"}
        if len(success)!=sum(r["status"]=="ok" for r in saved):
            raise ValueError("Duplicate successful response IDs")
        request_lookup={r["id"]:r for r in requests}
        for ident,record in success.items():
            if ident not in request_lookup or record["request_hash"]!=object_hash(request_lookup[ident]["request"]):
                raise ValueError("Saved response does not match prepared request")
            validate_answer(record["response"])
        attempted_ids={e["id"] for e in events if e["event"]=="started"}
        unfinished=attempted_ids-set(success)
        if unfinished and not retry_failed:
            raise ValueError("Failed/interrupted calls exist. Inspect the logs, then use --retry-failed to explicitly reattempt. Previous uncertain cost remains reserved.")
        pending=[r for r in requests if r["id"] not in success]
        key=None
        called=0
        status="complete" if not pending else "running"
        started_run=time.perf_counter()
        for item in pending:
            if max_requests is not None and called>=max_requests:
                status="request_limit"
                break
            used=budget_used(events)
            if used["accounted_usd"]+reserve>budget+1e-12:
                status="budget_limit"
                break
            if key is None:
                key=resolve_key()
                if not key.strip():
                    raise ValueError("An API key is required for --execute")
                started_run=time.perf_counter()  # Exclude local key-entry time.
            ident=uuid.uuid4().hex
            event={"event":"started","attempt_id":ident,"id":item["id"],"reserved_usd":reserve,
                   "started_utc":datetime.now(timezone.utc).isoformat(),"request_hash":object_hash(item["request"])}
            append(journal,event);events.append(event)
            called+=1
            response=None
            try:
                response,elapsed=fetcher(item["request"],key,timeout)
                answer=validate_answer(response)
                if response["usage"]["input_tokens"]>RESERVED_TOKENS_PER_CALL:
                    raise ValueError("Provider input-token usage exceeded the documented reservation bound")
                record={"id":item["id"],"split":item["split"],"status":"ok","attempt_id":ident,
                        "request_hash":event["request_hash"],"latency_ms":elapsed,"response":response}
                append(responses_path,record)
                cost=response["usage"]["input_tokens"]*PRICE_PER_MILLION/1_000_000
                done={"event":"finished","attempt_id":ident,"id":item["id"],"status":"ok","actual_cost_usd":cost}
                append(journal,done);events.append(done)
                success[item["id"]]=record
            except BaseException as error:
                # Never print provider error bodies, headers or the key.
                code=error.code if isinstance(error,urllib.error.HTTPError) else None
                record={"id":item["id"],"split":item["split"],"status":"failed","attempt_id":ident,
                        "request_hash":event["request_hash"],"error_type":type(error).__name__,"http_status":code}
                if response is not None:
                    record["response"]=response
                append(responses_path,record)
                done={"event":"finished","attempt_id":ident,"id":item["id"],"status":"failed","actual_cost_usd":None}
                append(journal,done);events.append(done)
                status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed"
                print(f"Stopped on {type(error).__name__}; HTTP status {code}. No automatic retry. See saved attempt ID {ident}.")
                break
            if called%25==0:
                print(f"Saved {len(success)}/{len(requests)} responses; accounted list-price/reserved cost ${budget_used(events)['accounted_usd']:.6f}",flush=True)
            if pause>0:
                time.sleep(pause)
        if len(success)==len(requests):
            status="complete"
        used=budget_used(events)
        summary={"status":status,"successes":len(success),"expected":len(requests),"calls_this_invocation":called,
                 "wall_seconds_this_invocation":time.perf_counter()-started_run,"budget_usd":budget,
                 "pause_seconds_between_calls":pause,"timeout_seconds":timeout,**used,
                 "note":"Budget accounting uses current list price and a conservative context-limit reservation. It is per run, not an account-wide credit cap or provider invoice."}
        (folder/"jev-run.json").write_text(json.dumps(summary,indent=2)+"\n")
        append(folder/"jev-invocations.jsonl",summary)
        print(json.dumps(summary,indent=2))
        return summary
    finally:
        lock.unlink(missing_ok=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--execute",action="store_true",help="Explicitly opt in to billable requests")
    parser.add_argument("--budget-usd",type=float,default=1.)
    parser.add_argument("--max-requests",type=int)
    parser.add_argument("--pause",type=float,default=.05)
    parser.add_argument("--timeout",type=float,default=30.)
    parser.add_argument("--retry-failed",action="store_true")
    args=parser.parse_args()
    if args.max_requests is not None and args.max_requests<1 or args.pause<0 or args.timeout<=0:
        parser.error("Use positive request count/timeout and nonnegative pause")
    outcome=run(args.run,args.budget_usd,args.execute,args.max_requests,args.pause,args.timeout,args.retry_failed)
    if args.execute and outcome["status"]!="complete":
        raise SystemExit(2)
