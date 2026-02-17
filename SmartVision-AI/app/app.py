# ...existing code...
import streamlit as st
from pathlib import Path
import importlib.util
import traceback

PAGES_DIR = Path(__file__).parent / "pages"

def _load_pages():
    pages = {}
    for f in sorted(PAGES_DIR.glob("*.py")):
        name = f.stem  # e.g. "1_Home"
        # sanitize module name so it never starts with digit
        mod_name = f"app.pages.{name}" if not name[0].isdigit() else f"app.pages._{name}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, str(f))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # human friendly title after first underscore if present
            title = name.split("_", 1)[1] if "_" in name else name
            pages[title] = mod
        except Exception:
            # ignore failing page imports but log to Streamlit
            pages[name] = None
    return pages

def _run_page(module):
    if module is None:
        st.error("Page failed to import. Check file for errors.")
        return
    for entry in ("run", "app", "main"):
        if hasattr(module, entry):
            try:
                getattr(module, entry)()
            except Exception as e:
                st.error(f"Error running page: {e}")
                st.text(traceback.format_exc())
            return
    st.warning("No entrypoint (run/app/main) found in page module. Add a run(app) function to the page.")

def main():
    st.set_page_config(page_title="SmartVision AI", layout="wide")
    st.sidebar.title("SmartVision AI")
    pages = _load_pages()
    page_names = list(pages.keys())
    choice = st.sidebar.selectbox("Choose a page", page_names)
    st.sidebar.markdown("---")
    st.title(choice)
    _run_page(pages[choice])

if __name__ == "__main__":
    main()
# ...existing code...