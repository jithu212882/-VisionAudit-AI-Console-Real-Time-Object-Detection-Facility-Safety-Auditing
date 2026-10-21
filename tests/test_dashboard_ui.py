"""
Frontend Dashboard UI Verification Tests.
Verifies DOM structure, HUD overlays, video viewport, connection controls, and accessibility elements in live_dashboard.html.
"""

from pathlib import Path


def test_dashboard_file_exists():
    html_file = Path("live_dashboard.html")
    assert html_file.exists(), "live_dashboard.html not found."


def test_dashboard_dom_elements():
    content = Path("live_dashboard.html").read_text(encoding="utf-8")

    # Essential stream controls
    assert 'id="wsUrl"' in content, "Missing wsUrl input."
    assert 'id="sourceInput"' in content, "Missing sourceInput input."
    assert 'id="previewCheckbox"' in content, "Missing previewCheckbox."
    assert 'id="connectBtn"' in content, "Missing connectBtn."
    assert 'preview=true' in content, "Default preview=true omitted from URL."

    # Video viewport and live feed
    assert 'id="videoViewport"' in content, "Missing videoViewport container."
    assert 'id="liveFeedImg"' in content, "Missing liveFeedImg element."
    assert 'id="feedPlaceholder"' in content, "Missing standby placeholder."

    # Live HUD overlay
    assert 'id="hudOverlay"' in content, "Missing hudOverlay container."
    assert 'id="overlayFps"' in content, "Missing overlayFps badge."
    assert 'id="overlayCount"' in content, "Missing overlayCount badge."
    assert 'id="overlayClasses"' in content, "Missing overlayClasses container."

    # Side panels
    assert 'id="occurrencesList"' in content, "Missing occurrencesList container."
    assert 'id="alertsStream"' in content, "Missing alertsStream container."

    # Keyboard shortcut
    assert 'Space' in content, "Keyboard shortcut handler missing."


if __name__ == "__main__":
    print("[*] Running Dashboard UI DOM Verification Tests...")
    test_dashboard_file_exists()
    test_dashboard_dom_elements()
    print("[+] All Dashboard UI tests passed successfully!")
