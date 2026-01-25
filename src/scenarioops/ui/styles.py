PIPELINE_CSS = """
<style>
.pipeline-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 6px 0 4px;
}
.pipeline-box {
  padding: 6px 10px;
  border-radius: 6px;
  border: 1px solid #d0d7de;
  background: #ffffff;
  font-size: 0.85rem;
  font-weight: 600;
  color: #111827;
}
.pipeline-box.running {
  background: #dbeafe;
  border-color: #93c5fd;
}
.pipeline-box.completed {
  background: #dcfce7;
  border-color: #86efac;
}
.pipeline-box.failed {
  background: #fee2e2;
  border-color: #fca5a5;
}
.pipeline-box.skipped {
  background: #f3f4f6;
  border-color: #e5e7eb;
  color: #6b7280;
}
.pipeline-box.pending {
  background: #f9fafb;
  border-color: #e5e7eb;
  color: #9ca3af;
}
.pipeline-label {
  margin: 4px 0 8px;
  font-size: 0.85rem;
  color: #4b5563;
}
</style>
"""
