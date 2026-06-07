import re
import subprocess
import json
import os
import time
import uuid
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import load_incidents, update_solution_success

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "data", "session_history.json")

ISSUE_CATEGORIES = {
    "password_login": {
        "label": "Password & Login",
        "icon": "🔑",
        "patterns": [
            r"password.*(?:correct|right|valid).*(?:error|not working|fail)",
            r"(?:can't|cannot|unable).*(?:login|sign in|log in)",
            r"(?:forgot|forget).*password",
            r"login.*(?:error|failed|not working|problem)",
            r"(?:session|logged out).*(?:expired|invalid|error)",
            r"two.?factor|2fa|mfa.*(?:not working|code|error)",
            r"authenticator.*(?:not working|code|wrong)",
        ],
        "complexity": "simple",
        "severity": "medium",
        "auto_fixable": True,
    },
    "website_app": {
        "label": "Website & App Issues",
        "icon": "🌐",
        "patterns": [
            r"website.*(?:not opening|down|unreachable|not loading|not accessible)",
            r"(?:can't|cannot|unable).*(?:access|reach|open).*(?:site|website|page)",
            r"(?:site|website|page).*(?:down|offline|error|not working)",
            r"(?:404|500|502|503|504).*error",
            r"app.*(?:crash|crashes|crashing|closing|not opening)",
            r"browser.*(?:crash|crashes|freeze|hangs|slow)",
        ],
        "complexity": "medium",
        "severity": "high",
        "auto_fixable": True,
    },
    "network_connection": {
        "label": "Network & Connection",
        "icon": "🔌",
        "patterns": [
            r"(?:connection|connect|network|internet).*(?:error|failed|refused|timeout|lost|dropped|disconnected)",
            r"(?:can't|cannot|unable).*(?:connect|reach)",
            r"(?:timeout|timed out).*(?:connection|request)",
            r"wifi.*(?:not working|disconnected|error|authentication|password)",
            r"vpn.*(?:not working|connect|error|failed)",
            r"bluetooth.*(?:not working|pair|connect|error)",
        ],
        "complexity": "medium",
        "severity": "high",
        "auto_fixable": True,
    },
    "email_communication": {
        "label": "Email Issues",
        "icon": "📧",
        "patterns": [
            r"email.*(?:not sending|not receiving|not arriving|delayed)",
            r"(?:can't|cannot|unable).*send.*email",
            r"email.*(?:error|failed|failure)",
            r"notification.*(?:not working|not sending|not receiving)",
        ],
        "complexity": "simple",
        "severity": "medium",
        "auto_fixable": False,
    },
    "performance": {
        "label": "Slow Performance",
        "icon": "🐌",
        "patterns": [
            r"(?:slow|lag|delay|hang|freeze|unresponsive|stuck)",
            r"(?:page|app|site|website).*(?:slow|taking too long|not responding)",
            r"(?:loading|load).*(?:slow|stuck|taking forever)",
            r"computer.*(?:slow|freeze|lag|unresponsive)",
            r"video.*(?:buffering|lag|not loading|freeze|stutter)",
            r"audio.*(?:not working|no sound|crackling|distorted)",
        ],
        "complexity": "complex",
        "severity": "low",
        "auto_fixable": False,
    },
    "hardware_device": {
        "label": "Hardware & Devices",
        "icon": "🖥️",
        "patterns": [
            r"printer.*(?:not working|error|offline|not responding)",
            r"mouse.*(?:not working|not responding|freeze)",
            r"keyboard.*(?:not working|keys|typing)",
            r"monitor.*(?:no display|blank|black screen|no signal)",
            r"usb.*(?:not working|not detected|error)",
            r"microphone.*(?:not working|no sound)", r"camera.*(?:not working|error)",
            r"speaker.*(?:not working|no sound)",
        ],
        "complexity": "complex",
        "severity": "medium",
        "auto_fixable": True,
    },
    "software_install": {
        "label": "Installation & Updates",
        "icon": "📦",
        "patterns": [
            r"(?:install|installation|setup).*(?:error|failed|not working|stuck)",
            r"update.*(?:error|failed|not working|stuck)",
            r"windows.*update.*(?:error|failed|not working)",
            r"app.*(?:install|installation).*(?:error|failed)",
            r"download.*(?:failed|error|slow|stuck|interrupted)",
        ],
        "complexity": "medium",
        "severity": "medium",
        "auto_fixable": True,
    },
    "file_disk": {
        "label": "Files & Storage",
        "icon": "💾",
        "patterns": [
            r"disk.*(?:full|space|error)",
            r"file.*(?:not opening|corrupted|deleted|missing|not found)",
            r"(?:out of|running out of).*(?:disk|storage|space)",
            r"backup.*(?:error|failed|not working)",
        ],
        "complexity": "medium",
        "severity": "high",
        "auto_fixable": True,
    },
    "security_privacy": {
        "label": "Security & Privacy",
        "icon": "🛡️",
        "patterns": [
            r"virus|malware|trojan|ransomware|spyware",
            r"antivirus.*(?:error|not working|blocking)",
            r"firewall.*(?:blocking|error|not working)",
            r"(?:hacked|compromised|breach|unauthorized)",
            r"pop.?up.*(?:ads|blocker|not working)",
            r"ad.?blocker.*(?:not working|error)",
        ],
        "complexity": "complex",
        "severity": "critical",
        "auto_fixable": False,
    },
    "payment": {
        "label": "Payment Issues",
        "icon": "💳",
        "patterns": [
            r"payment.*(?:error|failed|not working|declined)",
            r"(?:can't|cannot|unable).*pay|checkout.*(?:error|failed)",
            r"transaction.*(?:failed|error|declined|pending)",
            r"refund.*(?:not received|error|pending)",
        ],
        "complexity": "medium",
        "severity": "critical",
        "auto_fixable": False,
    },
    "display_visual": {
        "label": "Display & Visual",
        "icon": "🖼️",
        "patterns": [
            r"screen.*(?:flicker|blank|black|no display|tear|glitch)",
            r"display.*(?:error|not working|flicker)",
            r"resolution.*(?:wrong|error|not working)",
            r"font.*(?:blurry|small|large|not displaying|corrupted)",
            r"dark.?mode.*(?:not working|error)",
        ],
        "complexity": "medium",
        "severity": "low",
        "auto_fixable": True,
    },
    "search": {
        "label": "Search Issues",
        "icon": "🔍",
        "patterns": [
            r"search.*(?:not working|no results|error|broken)",
            r"(?:can't|cannot).*search|find.*(?:not working|error)",
        ],
        "complexity": "simple",
        "severity": "low",
        "auto_fixable": False,
    },
    "sync_backup": {
        "label": "Sync & Backup",
        "icon": "🔄",
        "patterns": [
            r"sync.*(?:not working|error|failed|stuck)",
            r"cloud.*(?:sync|not syncing|error)",
            r"backup.*(?:not working|failed|error)",
            r"onedrive|dropbox|google.?drive.*(?:not syncing|error)",
        ],
        "complexity": "medium",
        "severity": "medium",
        "auto_fixable": True,
    },
    "browser_ext": {
        "label": "Browser & Extensions",
        "icon": "🌍",
        "patterns": [
            r"extension.*(?:not working|error|crash|conflict)",
            r"ad.?block|ublock|adblock.*(?:not working|blocking)",
            r"bookmark.*(?:not syncing|deleted|missing)",
            r"tab.*(?:crash|crashing|closing|not loading)",
        ],
        "complexity": "simple",
        "severity": "low",
        "auto_fixable": True,
    },
    "server_infrastructure": {
        "label": "Server & Infrastructure",
        "icon": "🖧",
        "patterns": [
            r"(?:server|service|daemon).*(?:down|not running|unreachable|crash|offline|error|failed)",
            r"(?:port).*(?:not open|blocked|in use|refused|error)",
            r"(?:load balancer|reverse proxy).*(?:error|down|not working|unhealthy)",
            r"(?:nginx|apache|iis).*(?:error|down|not working|restart|config)",
            r"(?:cpu|memory|ram|disk).*(?:high|full|pegged|100%|exhausted)",
            r"(?:database|db).*(?:down|not connecting|connection refused|error|replication)",
            r"(?:docker|container|pod).*(?:crash|error|not starting|restarting|oom)",
            r"(?:ssl|tls|certificate).*(?:expired|error|invalid|mismatch|renew)",
            r"(?:cron|scheduler|scheduled task).*(?:not running|failed|error)",
            r"(?:deployment|release|rollout).*(?:failed|error|rollback|stuck)",
            r"(?:api endpoint|rest api|graphql).*(?:down|error|timeout|not responding)",
            r"(?:latency|response time|p95|p99).*(?:high|spike|increasing)",
            r"(?:health check|heartbeat).*(?:failing|error|down)",
            r"(?:backup|snapshot|restore).*(?:failed|error|corrupted|stale)",
        ],
        "complexity": "complex",
        "severity": "critical",
        "auto_fixable": True,
    },
    "system_generic": {
        "label": "General Troubleshoot",
        "icon": "🔧",
        "patterns": [
            r".*",
        ],
        "complexity": "medium",
        "severity": "medium",
        "auto_fixable": True,
        "catch_all": True,
    },
}

INTERACTIVE_FLOWS = {
    "password_login": {
        "greeting": "Let me help you with your login issue. I'll walk through this step-by-step.",
        "steps": [
            {"id": "clear_cache", "question": "Step 1: Clear your browser cache and cookies (Ctrl+Shift+Del). Did that fix it?", "auto_fix": None},
            {"id": "incognito", "question": "Step 2: Try incognito/private mode. This disables extensions that might interfere. Working now?", "auto_fix": None},
            {"id": "caps_check", "question": "Step 3: Check if Caps Lock is ON and your keyboard layout is correct. Try typing your password in a text editor first.", "auto_fix": None},
            {"id": "reset_pw", "question": "Step 4: Use 'Forgot Password' to reset. Check your email (including spam) for the reset link. Did you get it?", "auto_fix": "check_email"},
            {"id": "wait_lockout", "question": "Step 5: Account may be temporarily locked. Wait 15 min, then try again.", "auto_fix": None},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
    "website_app": {
        "greeting": "I'll check if the site is accessible and guide you through fixes.",
        "steps": [
            {"id": "dns_check", "question": "Step 1: I'll check DNS resolution. Try the site in a different browser meanwhile.", "auto_fix": "dns_lookup"},
            {"id": "ping_test", "question": "Step 2: Server is reachable. Have you tried restarting your router?", "auto_fix": "ping_test"},
            {"id": "other_sites", "question": "Step 3: Are OTHER websites working? If not, it's likely your internet connection.", "auto_fix": None},
            {"id": "vpn_disable", "question": "Step 4: Disable any VPN or proxy services, then try again.", "auto_fix": None},
            {"id": "dns_flush", "question": "Step 5: I can flush your DNS cache. Click 'Auto Fix' to run it.", "auto_fix": "flush_dns"},
            {"id": "browser_reset", "question": "Step 6: Reset your browser settings or try a different browser.", "auto_fix": None},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
    "network_connection": {
        "greeting": "Let me diagnose your network. I can run some checks and fix common issues.",
        "steps": [
            {"id": "ping_gateway", "question": "Step 1: Checking internet connectivity...", "auto_fix": "ping_gateway"},
            {"id": "router_reset", "question": "Step 2: Unplug your router for 30 seconds and plug it back in. Did that help?", "auto_fix": None},
            {"id": "dns_flush", "question": "Step 3: I can flush your DNS cache. Click 'Auto Fix' to run it.", "auto_fix": "flush_dns"},
            {"id": "network_reset", "question": "Step 4: I can reset your network stack. Click 'Auto Fix' (requires admin).", "auto_fix": "network_reset"},
            {"id": "other_devices", "question": "Step 5: Do other devices on your network work? If yes, it's your device. If no, it's your ISP/router.", "auto_fix": None},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
    "email_communication": {
        "greeting": "Let me help you troubleshoot your email issue.",
        "steps": [
            {"id": "spam_check", "question": "Step 1: Check your spam/junk folder. The email might be there. Found it?", "auto_fix": None},
            {"id": "address_check", "question": "Step 2: Verify the recipient's address - even one wrong character causes failure.", "auto_fix": None},
            {"id": "storage_check", "question": "Step 3: Is your mailbox full? Delete old emails if needed.", "auto_fix": None},
            {"id": "test_send", "question": "Step 4: Send a test email to yourself. If it arrives, the problem is on the recipient's side.", "auto_fix": None},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
    "performance": {
        "greeting": "Let me identify what's causing the slowdown.",
        "steps": [
            {"id": "restart", "question": "Step 1: Have you tried restarting your computer? This fixes most performance issues.", "auto_fix": None},
            {"id": "cache_clear", "question": "Step 2: Clear your browser cache and close unused tabs/programs.", "auto_fix": None},
            {"id": "disk_cleanup", "question": "Step 3: I can run disk cleanup to free up space. Click 'Auto Fix'.", "auto_fix": "disk_cleanup"},
            {"id": "startup_check", "question": "Step 4: Check Task Manager (Ctrl+Shift+Esc) for programs using high CPU/memory.", "auto_fix": None},
            {"id": "extensions", "question": "Step 5: Disable browser extensions. Some cause significant slowdowns.", "auto_fix": None},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
    "hardware_device": {
        "greeting": "Let me help you troubleshoot your hardware issue.",
        "steps": [
            {"id": "restart_device", "question": "Step 1: Restart your computer. This resolves many hardware detection issues.", "auto_fix": None},
            {"id": "check_cable", "question": "Step 2: Check all cable connections. Unplug and replug the device.", "auto_fix": None},
            {"id": "driver_check", "question": "Step 3: Check Device Manager for driver issues (yellow exclamation marks).", "auto_fix": "check_drivers"},
            {"id": "power_cycle", "question": "Step 4: Power cycle the device: turn off, wait 30 sec, turn on.", "auto_fix": None},
            {"id": "usb_port", "question": "Step 5: Try a different USB port or a different computer to isolate the issue.", "auto_fix": None},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
    "software_install": {
        "greeting": "I'll help you with the installation issue.",
        "steps": [
            {"id": "admin_run", "question": "Step 1: Try running the installer as Administrator (right-click → Run as Admin).", "auto_fix": None},
            {"id": "disk_space", "question": "Step 2: Check if you have enough disk space. I can check for you.", "auto_fix": "check_disk"},
            {"id": "antivirus", "question": "Step 3: Temporarily disable antivirus - it might be blocking the installation.", "auto_fix": None},
            {"id": "clean_temp", "question": "Step 4: Clean temporary files that might interfere. Click 'Auto Fix'.", "auto_fix": "clean_temp"},
            {"id": "compat_mode", "question": "Step 5: Try running in compatibility mode for an older Windows version.", "auto_fix": None},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
    "file_disk": {
        "greeting": "Let me help you with your storage/file issue.",
        "steps": [
            {"id": "disk_check", "question": "Step 1: I'll check your disk space. Click 'Auto Fix'.", "auto_fix": "check_disk"},
            {"id": "cleanup", "question": "Step 2: I can run Disk Cleanup to free space. Click 'Auto Fix'.", "auto_fix": "disk_cleanup"},
            {"id": "recycle_bin", "question": "Step 3: Empty your Recycle Bin and delete temporary files.", "auto_fix": None},
            {"id": "large_files", "question": "Step 4: Check for large files using 'Settings → Storage'.", "auto_fix": None},
            {"id": "error_check", "question": "Step 5: I can check the disk for errors. Click 'Auto Fix'.", "auto_fix": "check_disk_errors"},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
    "display_visual": {
        "greeting": "Let me help you fix your display issue.",
        "steps": [
            {"id": "cable_check", "question": "Step 1: Check monitor cable connections. Try reseating the cable.", "auto_fix": None},
            {"id": "display_settings", "question": "Step 2: Press Win+P and try switching display modes (Duplicate/Extend).", "auto_fix": None},
            {"id": "resolution", "question": "Step 3: Right-click desktop → Display Settings → Try a different resolution.", "auto_fix": None},
            {"id": "driver_update", "question": "Step 4: Update your graphics driver from Device Manager.", "auto_fix": "check_drivers"},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
    "browser_ext": {
        "greeting": "Let me help you with your browser issue.",
        "steps": [
            {"id": "restart_browser", "question": "Step 1: Close and reopen your browser. Still happening?", "auto_fix": None},
            {"id": "incognito", "question": "Step 2: Try incognito mode. If it works, an extension is the problem.", "auto_fix": None},
            {"id": "disable_ext", "question": "Step 3: Disable all extensions, then re-enable one by one to find the culprit.", "auto_fix": None},
            {"id": "clear_cache", "question": "Step 4: Clear browsing data (cache, cookies).", "auto_fix": None},
            {"id": "reset_browser", "question": "Step 5: Reset browser to default settings.", "auto_fix": None},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
    "sync_backup": {
        "greeting": "Let me help you with your sync/backup issue.",
        "steps": [
            {"id": "restart_app", "question": "Step 1: Close and reopen the sync app. Often fixes temporary glitches.", "auto_fix": None},
            {"id": "check_login", "question": "Step 2: Make sure you're logged in to your account across all devices.", "auto_fix": None},
            {"id": "check_connection", "question": "Step 3: Check your internet connection. Sync requires stable internet.", "auto_fix": "ping_gateway"},
            {"id": "pause_resume", "question": "Step 4: Pause and resume syncing in the app settings.", "auto_fix": None},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
    "server_infrastructure": {
        "greeting": "Let me diagnose your server issue. I'll run remote checks and guide you through fixes.",
        "steps": [
            {"id": "ping_server", "question": "Step 1: Checking if the server is reachable...", "auto_fix": "ping_test"},
            {"id": "dns_resolve", "question": "Step 2: Checking DNS resolution for the server...", "auto_fix": "dns_lookup"},
            {"id": "port_check", "question": "Step 3: I'll check common ports. Is the server on your local network or remote?", "auto_fix": "port_scan"},
            {"id": "service_check", "question": "Step 4: Check if the service/process is running. Try 'systemctl status <service>' or check Task Manager.", "auto_fix": "check_services"},
            {"id": "ssl_check", "question": "Step 5: Check SSL certificate expiry and validity.", "auto_fix": None},
            {"id": "disk_check", "question": "Step 6: Check if server disk is full. I can check disk space.", "auto_fix": "check_disk"},
            {"id": "logs_check", "question": "Step 7: Check server logs for errors (typically in /var/log/ or Event Viewer).", "auto_fix": None},
            {"id": "restart_service", "question": "Step 8: Try restarting the service. Click Auto-Fix to attempt.", "auto_fix": None},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
    "system_generic": {
        "greeting": "I'll run a full system diagnostic to identify the issue.",
        "steps": [
            {"id": "full_scan", "question": "Step 1: Running comprehensive system scan... Click Auto-Fix to check your system health.", "auto_fix": "full_scan"},
            {"id": "check_connectivity", "question": "Step 2: Checking internet connectivity and DNS...", "auto_fix": "ping_gateway"},
            {"id": "check_disk", "question": "Step 3: Checking disk space and health...", "auto_fix": "check_disk"},
            {"id": "check_memory", "question": "Step 4: Check if your system has enough memory. Close unused programs.", "auto_fix": None},
            {"id": "check_drivers", "question": "Step 5: Checking for driver issues...", "auto_fix": "check_drivers"},
            {"id": "generic_advice", "question": "Step 6: Restart your computer if you haven't already. This fixes many issues.", "auto_fix": None},
            {"id": "escalate", "question": None, "auto_fix": None},
        ],
    },
}

AUTO_FIX_COMMANDS = {
    "dns_lookup": {"cmd": "nslookup google.com 2>&1", "desc": "DNS Resolution Check", "admin": False},
    "ping_test": {"cmd": "ping -n 4 google.com 2>&1", "desc": "Connectivity Test", "admin": False},
    "ping_gateway": {"cmd": "ping -n 2 8.8.8.8 2>&1", "desc": "Internet Gateway Test", "admin": False},
    "flush_dns": {"cmd": "ipconfig /flushdns 2>&1", "desc": "DNS Cache Flush", "admin": True},
    "network_reset": {"cmd": "netsh int ip reset 2>&1 & netsh winsock reset 2>&1", "desc": "Network Stack Reset", "admin": True},
    "disk_cleanup": {"cmd": "cleanmgr /sagerun:1 2>&1", "desc": "Disk Cleanup", "admin": True},
    "check_disk": {"cmd": "wmic logicaldisk get size,freespace,caption 2>&1", "desc": "Disk Space Check", "admin": False},
    "check_disk_errors": {"cmd": "chkdsk /f 2>&1", "desc": "Disk Error Check", "admin": True},
    "clean_temp": {"cmd": "del /q /s %temp%\\* 2>nul & del /q /s C:\\Windows\\Temp\\* 2>nul", "desc": "Temp Files Cleanup", "admin": True},
    "check_drivers": {"cmd": "pnputil /enum-drivers 2>&1 | findstr -i \"Problem\" 2>&1", "desc": "Driver Health Check", "admin": False},
    "check_email": {"cmd": "ping smtp.gmail.com -n 1 2>&1", "desc": "Email Server Check", "admin": False},
    "port_scan": {"cmd": "powershell -Command \"Get-NetTCPConnection -State Listen | Select-Object -First 10 2>&1\" 2>&1", "desc": "Active Ports Scan", "admin": False},
    "check_services": {"cmd": "powershell -Command \"Get-Service | Where-Object {$_.Status -eq 'Stopped' -and $_.StartType -eq 'Automatic'} | Select-Object -First 10 Name,Status 2>&1\" 2>&1", "desc": "Stopped Services Check", "admin": False},
    "full_scan": {"cmd": "powershell -Command \"Write-Host '=== System Info ==='; Get-ComputerInfo -Property OsName,OsVersion,OsArchitecture; Write-Host '=== Disk ==='; Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free | Format-Table -AutoSize; Write-Host '=== Memory ==='; Get-CimInstance Win32_OperatingSystem | Select-Object @{Name='TotalGB';Expression={[math]::Round($_.TotalVisibleMemorySize/1MB,1)}},@{Name='FreeGB';Expression={[math]::Round($_.FreePhysicalMemory/1MB,1)}} 2>&1\"", "desc": "Full System Scan", "admin": False},
    "restart_self": {"cmd": "powershell -Command \"Write-Host 'Auto-restart not available in this environment. Please restart manually.'\"", "desc": "Restart Service", "admin": False},
}

ROOT_CAUSE_PROBABILITIES = {
    "password_login": [
        {"cause": "Browser cache/cookies corruption", "probability": 0.35, "fix": "Clear cache and cookies"},
        {"cause": "Browser extension interference", "probability": 0.20, "fix": "Use incognito mode"},
        {"cause": "Account lockout due to multiple attempts", "probability": 0.18, "fix": "Wait 15 minutes"},
        {"cause": "Password expired or needs reset", "probability": 0.15, "fix": "Use forgot password flow"},
        {"cause": "Browser compatibility issue", "probability": 0.12, "fix": "Try a different browser"},
    ],
    "website_app": [
        {"cause": "Local DNS cache issue", "probability": 0.30, "fix": "Flush DNS cache"},
        {"cause": "Internet connectivity problem", "probability": 0.25, "fix": "Check internet connection"},
        {"cause": "Website server outage", "probability": 0.20, "fix": "Use downdetector to verify"},
        {"cause": "VPN/Proxy interference", "probability": 0.15, "fix": "Disable VPN/proxy"},
        {"cause": "Browser cache corruption", "probability": 0.10, "fix": "Clear browser cache"},
    ],
    "network_connection": [
        {"cause": "DNS resolution failure", "probability": 0.30, "fix": "Flush DNS and use Google DNS"},
        {"cause": "Router/modem needs restart", "probability": 0.25, "fix": "Power cycle router"},
        {"cause": "Network adapter driver issue", "probability": 0.20, "fix": "Update or reinstall network driver"},
        {"cause": "ISP outage in your area", "probability": 0.15, "fix": "Check with ISP"},
        {"cause": "Firewall/security software blocking", "probability": 0.10, "fix": "Temporarily disable firewall"},
    ],
    "email_communication": [
        {"cause": "Email in spam/junk folder", "probability": 0.40, "fix": "Check spam folder"},
        {"cause": "Incorrect email address", "probability": 0.25, "fix": "Verify recipient address"},
        {"cause": "Mailbox full", "probability": 0.20, "fix": "Delete old emails"},
        {"cause": "Email server temporarily down", "probability": 0.15, "fix": "Wait and try again"},
    ],
    "performance": [
        {"cause": "Too many programs running", "probability": 0.30, "fix": "Close unused programs"},
        {"cause": "Browser cache buildup", "probability": 0.25, "fix": "Clear browser cache"},
        {"cause": "Low disk space", "probability": 0.20, "fix": "Free up disk space"},
        {"cause": "Browser extensions slowing down", "probability": 0.15, "fix": "Disable extensions"},
        {"cause": "Background updates running", "probability": 0.10, "fix": "Check for pending updates"},
    ],
    "server_infrastructure": [
        {"cause": "Service/process crashed or not running", "probability": 0.30, "fix": "Restart the service"},
        {"cause": "Disk space full on server", "probability": 0.20, "fix": "Free up server disk space"},
        {"cause": "SSL certificate expired", "probability": 0.15, "fix": "Renew SSL certificate"},
        {"cause": "Memory/CPU exhausted", "probability": 0.15, "fix": "Scale up or optimize"},
        {"cause": "Network/firewall blocking access", "probability": 0.10, "fix": "Check firewall rules"},
        {"cause": "Database connection pool full", "probability": 0.10, "fix": "Increase pool size or kill idle connections"},
    ],
    "system_generic": [
        {"cause": "Temporary system glitch", "probability": 0.30, "fix": "Restart your computer"},
        {"cause": "Background application conflict", "probability": 0.20, "fix": "Close unused applications"},
        {"cause": "Driver or update issue", "probability": 0.18, "fix": "Check for driver updates"},
        {"cause": "Disk space or memory low", "probability": 0.17, "fix": "Free up system resources"},
        {"cause": "Corrupted system cache", "probability": 0.15, "fix": "Clear system cache"},
    ],
}

COMMON_FIXES_MAP = {
    "password_login": ["Clear browser cache (Ctrl+Shift+Del)", "Try incognito mode", "Check Caps Lock", "Use Forgot Password flow", "Try a different browser"],
    "website_app": ["Check internet connection", "Restart router", "Try a different browser", "Clear DNS cache (ipconfig /flushdns)", "Disable VPN/proxy"],
    "network_connection": ["Restart router/modem", "Flush DNS (ipconfig /flushdns)", "Run network troubleshooter", "Update network driver", "Check cable connections"],
    "email_communication": ["Check spam folder", "Verify recipient address", "Check mailbox storage", "Try sending a test email"],
    "performance": ["Restart your computer", "Close unused programs/tabs", "Clear browser cache", "Check disk space", "Disable startup programs"],
    "hardware_device": ["Restart computer", "Check cable connections", "Try a different USB port", "Update drivers", "Check Device Manager"],
    "software_install": ["Run as Administrator", "Disable antivirus temporarily", "Check disk space", "Clean temporary files", "Run compatibility troubleshooter"],
    "file_disk": ["Empty Recycle Bin", "Run Disk Cleanup", "Delete temporary files", "Uninstall unused programs", "Move files to external drive"],
    "security_privacy": ["Run full antivirus scan", "Update all software", "Change passwords", "Enable firewall", "Check recent account activity"],
    "payment": ["Check card details", "Try a different payment method", "Contact bank", "Clear browser cache", "Try incognito mode"],
    "display_visual": ["Check cable connections", "Press Win+P to switch modes", "Update graphics driver", "Check display settings", "Try a different monitor"],
    "search": ["Clear browser cache", "Try a different search engine", "Check internet connection", "Disable browser extensions"],
    "sync_backup": ["Restart the sync app", "Check internet connection", "Sign out and sign back in", "Pause and resume sync", "Check storage quota"],
    "browser_ext": ["Restart browser", "Try incognito mode", "Disable all extensions", "Clear browser cache", "Reset browser settings"],
    "server_infrastructure": ["Restart the service/process", "Check server logs", "Free up disk space", "Renew SSL certificates", "Check firewall rules", "Restart the server if needed", "Verify DNS records", "Check database connectivity"],
    "system_generic": ["Restart your computer", "Run a virus scan", "Check for Windows updates", "Clear temporary files", "Run System File Checker (sfc /scannow)", "Check disk for errors (chkdsk)", "Update device drivers"],
    "default": ["Restart your computer", "Check for updates", "Run as Administrator", "Check error logs", "Search for specific error message"],
}


class TroubleshootingSession:
    def __init__(self, issue_type: str, description: str = ""):
        self.id = str(uuid.uuid4())[:8]
        self.issue_type = issue_type
        self.description = description
        self.start_time = time.time()
        self.steps = INTERACTIVE_FLOWS.get(issue_type, {}).get("steps", [])
        self.current_step = 0
        self.greeting = INTERACTIVE_FLOWS.get(issue_type, {}).get("greeting", "")
        self.results = []
        self.resolved = False
        self.escalated = False

    def is_done(self):
        return self.current_step >= len(self.steps) or self.resolved

    def current(self):
        if self.is_done():
            return None
        return self.steps[self.current_step]

    def advance(self, worked: bool):
        step = self.current()
        if step:
            self.results.append({"step": step["id"], "question": step.get("question", ""), "worked": worked})
        if worked:
            self.resolved = True
        else:
            self.current_step += 1

    def get_progress(self):
        total = len(self.steps)
        done = self.current_step if self.resolved else self.current_step
        return {"current": min(done + 1, total), "total": total, "percent": round((done / total) * 100) if total > 0 else 0}

    def get_duration(self):
        return round(time.time() - self.start_time)

    def summary(self):
        return {
            "id": self.id,
            "issue_type": self.issue_type,
            "description": self.description,
            "duration_seconds": self.get_duration(),
            "resolved": self.resolved,
            "escalated": self.escalated,
            "steps_attempted": len(self.results),
            "steps": self.results,
        }


class IncidentEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
        self.incidents = load_incidents()
        self.sessions = {}
        self._ensure_history()
        self._build_index()

    def _ensure_history(self):
        if not os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "w") as f:
                json.dump([], f)

    def _load_history(self):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)

    def _save_history(self, history):
        with open(HISTORY_PATH, "w") as f:
            json.dump(history[-50:], f, indent=2)

    def _build_index(self):
        texts = [f"{i['title']} {i['description']} {' '.join(i.get('tags', []))}" for i in self.incidents]
        if texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            self.tfidf_matrix = None

    def refresh(self):
        self.incidents = load_incidents()
        self._build_index()

    def find_similar_incidents(self, query: str, top_k: int = 5):
        self.refresh()
        if not self.incidents or self.tfidf_matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [{"incident": self.incidents[idx], "similarity": float(round(similarities[idx], 3))} for idx in top_indices if similarities[idx] > 0.05]

    def group_incidents(self):
        self.refresh()
        groups, assigned = {}, set()
        for i, inc in enumerate(self.incidents):
            if i in assigned:
                continue
            tags = set(inc.get("tags", []))
            group = {"id": inc["id"], "title": inc["title"], "severity": inc["severity"], "timestamp": inc["timestamp"], "members": [inc]}
            assigned.add(i)
            for j, other in enumerate(self.incidents):
                if j in assigned:
                    continue
                if len(tags & set(other.get("tags", []))) >= 2:
                    group["members"].append(other)
                    assigned.add(j)
            key = "_".join(sorted(inc.get("tags", []))[:2])
            groups.setdefault(key, group)["members"].extend(group["members"]) if key in groups else (groups.update({key: group}))
        return list(groups.values())

    def classify_issue(self, query: str):
        query_lower = query.lower()
        for issue_type, info in ISSUE_CATEGORIES.items():
            for pattern in info["patterns"]:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return {"matched": True, "issue_type": issue_type, "info": info}
        return {"matched": False}

    def get_root_cause_probabilities(self, issue_type: str):
        return ROOT_CAUSE_PROBABILITIES.get(issue_type, [])

    def get_common_fixes(self, issue_type: str):
        return COMMON_FIXES_MAP.get(issue_type, COMMON_FIXES_MAP["default"])

    def run_auto_fix(self, fix_id: str):
        cmd_info = AUTO_FIX_COMMANDS.get(fix_id)
        if not cmd_info:
            return {"status": "unavailable", "message": "Fix not available.", "fix_name": "Unknown"}
        try:
            result = subprocess.run(["powershell", "-Command", cmd_info["cmd"]], capture_output=True, text=True, timeout=30)
            success = result.returncode == 0
            output = (result.stdout or result.stderr)[:300]
            return {"status": "success" if success else "error", "output": output, "fix_name": cmd_info["desc"], "admin_required": cmd_info["admin"]}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "message": "Command timed out.", "fix_name": cmd_info["desc"]}
        except Exception as e:
            return {"status": "error", "message": str(e), "fix_name": cmd_info["desc"]}

    def run_system_scan(self):
        results = {}
        for fix_id in ["dns_lookup", "ping_gateway", "check_disk", "check_drivers"]:
            results[fix_id] = self.run_auto_fix(fix_id)
        return results

    def start_session(self, session_id: str, issue_type: str, description: str = ""):
        self.sessions[session_id] = TroubleshootingSession(issue_type, description)
        return self.sessions[session_id]

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    def end_session(self, session_id: str):
        session = self.sessions.pop(session_id, None)
        if session:
            history = self._load_history()
            history.append(session.summary())
            self._save_history(history)
        return session.summary() if session else None

    def get_session_history(self):
        return self._load_history()

    def get_external_references(self, incident: dict):
        refs = {
            "database": [{"title": "MySQL Connection Pool Best Practices", "url": "https://dev.mysql.com/doc/refman/8.0/en/connection-pool-tuning.html", "source": "MySQL Docs"}],
            "ssl": [{"title": "Let's Encrypt Certificate Automation", "url": "https://letsencrypt.org/docs/", "source": "Let's Encrypt"}],
            "ddos": [{"title": "AWS Shield Advanced DDoS Protection", "url": "https://aws.amazon.com/shield/", "source": "AWS"}],
            "memory": [{"title": "Fixing Memory Leaks in Python Applications", "url": "https://pypi.org/project/memray/", "source": "Python"}],
            "redis": [{"title": "Redis Persistence and High Availability", "url": "https://redis.io/topics/persistence", "source": "Redis Docs"}],
            "kubernetes": [{"title": "Kubernetes Production Best Practices", "url": "https://kubernetes.io/docs/setup/best-practices/", "source": "K8s Docs"}],
            "cdn": [{"title": "CDN Caching Best Practices", "url": "https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/caching.html", "source": "AWS CloudFront"}],
        }
        return [ref for tag in incident.get("tags", []) for ref in refs.get(tag, [])][:5]

    def get_prevention_recommendations(self, incident: dict):
        recs = []
        tags = incident.get("tags", [])
        similar = self.find_similar_incidents(f"{incident['title']} {' '.join(tags)}", top_k=5)
        if len(similar) >= 2:
            recs.append("This issue type has occurred multiple times. Set up automated monitoring alerts to detect early warning signs.")
        if "monitoring" not in " ".join(incident.get("resolution", "")).lower():
            recs.append("Add monitoring and alerting for this component to detect similar issues before they impact users.")
        if "database" in tags:
            recs.append("Implement database connection pooling with proper limits and monitoring at 80% capacity.")
        if "security" in tags:
            recs.append("Schedule regular security audits and automated compliance scanning.")
        if "backup" not in str(incident.get("resolution", "")).lower():
            recs.append("Verify backup strategy includes automated testing of restore procedures.")
        recs.append("Create/update runbook documentation for this incident type with step-by-step resolution guide.")
        recs.append("Schedule a post-mortem meeting to identify systemic improvements.")
        return recs[:5]

    def record_solution_outcome(self, incident_id: int, solution_index: int, success: bool):
        update_solution_success(incident_id, solution_index, success)
        self.refresh()
