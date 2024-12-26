import json
import logging


class JSONFormatter(logging.Formatter):
    def format(self, record):
        record_dict = {
            "timestamp": record.created,
            "event": getattr(record, "event", "unknown"),
            "problem_id": getattr(record, "problem_id", None),
            "parent_id": getattr(record, "parent_id", None),
            "depth": getattr(record, "depth", 0),
            "sender_id": getattr(record, "sender_id", "unknown"),
            "receiver_id": getattr(record, "receiver_id", "unknown"),
            "content": getattr(record, "content", {}),
        }
        return json.dumps(record_dict)


class HierarchicalLoggingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = {}

    def emit(self, record):
        problem_id = getattr(record, "problem_id", None)
        if problem_id is None:
            # Skip logs without a problem_id
            return

        parent_id = getattr(record, "parent_id", None)
        depth = getattr(record, "depth", 0)

        log_entry = {
            "timestamp": record.created,
            "event": getattr(record, "event", "unknown"),
            "sender_id": getattr(record, "sender_id", "unknown"),
            "receiver_id": getattr(record, "receiver_id", "unknown"),
            "content": getattr(record, "content", {}),
        }

        self._add_log_entry(problem_id, parent_id, depth, log_entry)

    def _add_log_entry(self, problem_id, parent_id, depth, log_entry):
        if problem_id not in self.logs:
            self.logs[problem_id] = {"parent_id": parent_id, "depth": depth, "events": [], "subproblems": {}}
        self.logs[problem_id]["events"].append(log_entry)

    def get_logs(self):
        # Link each node with its parent
        all_nodes = self.logs
        for pid, node in all_nodes.items():
            parent_id = node["parent_id"]
            if parent_id and parent_id in all_nodes:
                all_nodes[parent_id]["subproblems"][pid] = node

        # Find root nodes
        roots = {pid: node for pid, node in all_nodes.items() if node["parent_id"] is None}
        return self._convert_sets_to_lists(roots)

    def _convert_sets_to_lists(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_sets_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_sets_to_lists(i) for i in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj


logger = logging.getLogger("awms")
handler = HierarchicalLoggingHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
