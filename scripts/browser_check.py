"""End-to-end checks against a running local, single-user playground."""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

from data.datasets import ROOT
from playground.service import Playground


def check_browser(url, workspace):
    for attempt in range(40):
        try:
            with urlopen(url + "/config", timeout=2) as response:
                assert response.status == 200
            break
        except Exception:
            if attempt == 39:
                raise
            time.sleep(1)
    playground = Playground(workspace)
    screenshots = ROOT / "docs" / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    errors, remote = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        context.on("request", lambda request: remote.append(request.url)
                   if urlparse(request.url).hostname not in {"localhost", "127.0.0.1", None} else None)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))

        def train():
            number = len(playground.pairs()) + 1
            page.get_by_role("button", name="Train and compare", exact=True).click()
            expect(page.locator("#simple-status")).to_contain_text(f"Saved Run {number}.", timeout=120000)
            expect(page.locator("#simple-summary")).to_contain_text(f"Run {number}", timeout=60000)
            expect(page.locator("#simple-metrics")).to_contain_text("Missed alerts", timeout=60000)
            print("UI saved Run", number, flush=True)
            return playground.pairs()[-1]

        try:
            page.goto(url)
            expect(page.get_by_role("button", name="Train and compare", exact=True)).to_be_visible()
            page.get_by_role("radio", name="Give missed alerts more importance", exact=True).check()
            first = train()
            page.screenshot(path=str(screenshots / "playground.png"), full_page=True)
            page.get_by_role("radio", name="Leave out flagged readings", exact=True).check()
            train()
            page.get_by_role("radio", name="Leave out location information", exact=True).check()
            train()

            row = page.locator("#simple-training-data").get_by_role("row").filter(has_text="air-0000")
            row.get_by_role("button", name="normal", exact=True).dblclick()
            expect(row.get_by_role("textbox", name="Edit cell", exact=True)).to_be_visible()
            row.get_by_role("textbox", name="Edit cell", exact=True).fill("alert")
            row.get_by_role("textbox", name="Edit cell", exact=True).press("Enter")
            expect(page.get_by_role("radio", name="Edit labels or include readings", exact=True)).to_be_checked(timeout=30000)
            last = train()
            _, changed = playground.records(last)
            assert changed["dataset"]["changes"]["changed_labels"] == 1
            page.screenshot(path=str(screenshots / "label-edit.png"), full_page=True)

            page.get_by_role("button", name="Download experiment record", exact=True).click()
            expect(page.locator("#simple-download")).to_be_visible(timeout=30000)
            assert playground.export(last["id"]).exists()
            with page.expect_download() as exported:
                page.locator("#simple-download").get_by_role("link").click()
            assert exported.value.suggested_filename == "experiment-record.zip"
            with page.expect_download() as downloaded:
                page.get_by_role("button", name="Source CSV", exact=True).click()
            assert downloaded.value.suggested_filename == "air_quality.csv"

            page.get_by_role("tab", name="Saved runs", exact=True).click()
            page.get_by_role("listbox", name="Comparison", exact=True).click()
            page.get_by_role("option", name=f"Run {first['number']}: {first['decision']}", exact=True).click()
            page.get_by_role("button", name="Open comparison", exact=True).click()
            expect(page.get_by_role("tab", name="Playground", exact=True)).to_have_attribute("aria-selected", "true", timeout=60000)
            expect(page.locator("#simple-summary")).to_contain_text(f"Run {first['number']}", timeout=60000)

            page.get_by_role("tab", name="Saved runs", exact=True).click()
            page.get_by_role("button", name="Refresh saved runs", exact=True).click()
            page.get_by_role("tab", name="Playground", exact=True).click()
            expect(page.locator("#simple-summary")).to_contain_text(f"Run {first['number']}")
            page.get_by_role("button", name="Download experiment record", exact=True).click()
            expect(page.locator("#simple-download")).to_be_visible(timeout=30000)
            with page.expect_download() as reopened:
                page.locator("#simple-download").get_by_role("link").click()
            from zipfile import ZipFile
            with ZipFile(reopened.value.path()) as archive:
                assert json.loads(archive.read("experiment.json"))["comparison"]["id"] == first["id"]

            page.get_by_role("button", name="Learned weights", exact=False).click()
            expect(page.get_by_role("tab", name="Your changed model", exact=True)).to_be_visible()
            page.get_by_role("tab", name="Your changed model", exact=True).click()
            page.screenshot(path=str(screenshots / "weights.png"), full_page=True)

            page.get_by_role("tab", name="Advanced", exact=True).click()
            expect(page.get_by_role("tab", name="Project & goals", exact=True)).to_be_visible()
            page.get_by_role("button", name="Refresh advanced experiments", exact=True).click()
            page.get_by_role("tab", name="Model weights", exact=True).click()
            expect(page.get_by_role("listbox", name="Neural checkpoint", exact=True)).to_be_visible()
            page.screenshot(path=str(screenshots / "advanced.png"), full_page=True)

            page.reload()
            expect(page.locator("#simple-summary")).to_contain_text(f"Run {last['number']}", timeout=60000)
            for width in [390, 768]:
                mobile = context.new_page()
                mobile.set_viewport_size({"width": width, "height": 844})
                mobile.goto(url)
                expect(mobile.locator("#simple-summary")).to_contain_text(f"Run {last['number']}", timeout=60000)
                assert mobile.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), f"Overflow at {width}px"
                if width < 700:
                    data_box = mobile.locator("#simple-training-data").bounding_box()
                    choice_box = mobile.locator("#simple-choice").bounding_box()
                    assert choice_box["y"] > data_box["y"] + data_box["height"], "Mobile controls must stack"
                    assert mobile.locator("#simple-metrics").bounding_box()["width"] > 300
                mobile.screenshot(path=str(screenshots / f"mobile-{width}.png"), full_page=True)
                mobile.close()
            assert not errors, errors
            assert not remote, remote
            print(json.dumps({"page_errors": errors, "external_requests": remote,
                              "saved_comparisons": len(playground.pairs()), "screenshots": str(screenshots)}, indent=2))
        except Exception:
            page.screenshot(path=str(ROOT / "outputs" / "qa" / "browser-failure.png"), full_page=True)
            print(page.locator("body").aria_snapshot()[-8000:])
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7861")
    parser.add_argument("--workspace", type=Path, default=ROOT / "outputs" / "simple-lab")
    args = parser.parse_args()
    check_browser(args.url, args.workspace)
