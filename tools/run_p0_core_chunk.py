import argparse
import faulthandler
import os
import sys
import traceback


def main():
    print("booting-p0-core", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--output_subdir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--cpu_threads", type=int, default=1)
    args = parser.parse_args()

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    sys.path.insert(0, PROJECT_ROOT)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.output_subdir:
        output_root = os.path.join(PROJECT_ROOT, "Output", args.output_subdir)
    else:
        output_root = os.path.join(PROJECT_ROOT, "Output", "revision_p0")
    os.makedirs(output_root, exist_ok=True)
    trace_path = os.path.join(output_root, "core_chunk_trace.txt")
    fault_path = os.path.join(output_root, "core_chunk_fault.log")
    fault_file = open(fault_path, "a", encoding="utf-8")
    faulthandler.enable(file=fault_file, all_threads=True)
    try:
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(f"datasets={datasets}; models={models}; seeds={seeds}; epochs={args.epochs}; patience={args.patience}\n")
            f.write("importing-revision-module\n")
        cpu_threads = max(1, int(args.cpu_threads))
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
        import torch
        from src.revision_p0_experiments import ensure_dir, run_core_holdout_stats

        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        output_root = ensure_dir(output_root)
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write("starting-run_core_holdout_stats\n")
        run_core_holdout_stats(
            output_root=output_root,
            device=torch.device("cpu"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            datasets=datasets,
            models=models,
            seeds=seeds,
        )
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write("chunk-done\n")
        print("done", flush=True)
    except Exception:
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write("chunk-exception\n")
            f.write(traceback.format_exc())
            if not traceback.format_exc().endswith("\n"):
                f.write("\n")
        raise
    finally:
        fault_file.flush()
        fault_file.close()


if __name__ == "__main__":
    main()
