#date: 2025-06-06T16:53:44Z
#url: https://api.github.com/gists/d74e57714024eac4041bdb96dafb007f
#owner: https://api.github.com/users/kordless

import sys
import os
import logging
import requests
import re
import json
from typing import Dict, Any, Optional, List, Tuple
from mcp.server.fastmcp import FastMCP, Context
from urllib.parse import urlparse, urljoin
import asyncio
import subprocess

__version__ = "0.1.1"
__updated__ = "2025-06-06"

# Setup logging
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
logs_dir = os.path.join(parent_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

log_file = os.path.join(logs_dir, "status_page_finder.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file)]
)
logger = logging.getLogger("status_page_finder")

# Initialize MCP server
mcp = FastMCP("status-page-finder-server")

# Common status page patterns and subdomains
STATUS_PAGE_PATTERNS = [
    # Subdomain patterns
    "status.",
    "uptime.",
    "health.",
    "system.",
    "monitor.",
    "statuspage.",
    "status-page.",
    "incidents.",
    # Path patterns
    "/status",
    "/uptime",
    "/health",
    "/system-status",
    "/statuspage",
    "/status-page",
    "/incidents",
    "/system",
]

# Known status page providers
STATUS_PAGE_PROVIDERS = {
    "statuspage.io": {
        "pattern": r"statuspage\.io",
        "indicators": ["powered by Atlassian Statuspage", "statuspage.io", "status.io"]
    },
    "betteruptime": {
        "pattern": r"betterstack|betteruptime",
        "indicators": ["Better Stack", "Better Uptime", "betteruptime.com"]
    },
    "cachet": {
        "pattern": r"cachet",
        "indicators": ["Cachet", "cachethq"]
    },
    "upptime": {
        "pattern": r"upptime",
        "indicators": ["Upptime", "Koj"]
    },
    "uptime-kuma": {
        "pattern": r"uptime-kuma",
        "indicators": ["Uptime Kuma", "louislam/uptime-kuma"]
    },
    "instatus": {
        "pattern": r"instatus",
        "indicators": ["Instatus", "instatus.com"]
    },
    "custom": {
        "pattern": r"",
        "indicators": []
    }
}

# Wraith crawler settings
WRAITH_LOCAL_URL = "http://localhost:5678/api"
WRAITH_REMOTE_URL = "https://wraith.nuts.services/api"

def guess_status_page_urls(domain: str) -> List[str]:
    """Generate potential status page URLs for a domain."""
    urls = []
    
    # Clean domain
    domain = domain.lower().strip()
    if domain.startswith("http://") or domain.startswith("https://"):
        parsed = urlparse(domain)
        domain = parsed.netloc
    
    # Remove www prefix if present
    if domain.startswith("www."):
        domain = domain[4:]
    
    # Generate subdomain variations
    for pattern in STATUS_PAGE_PATTERNS:
        if pattern.startswith("/"):
            # Path pattern
            urls.append(f"https://{domain}{pattern}")
            urls.append(f"https://www.{domain}{pattern}")
        elif pattern.endswith("."):
            # Subdomain pattern
            urls.append(f"https://{pattern}{domain}")
    
    # Add the domain itself (might redirect to status)
    urls.append(f"https://{domain}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls

async def crawl_with_wraith(url: str, use_local: bool = False) -> Dict[str, Any]:
    """Use Wraith crawler to get page content."""
    try:
        api_url = WRAITH_LOCAL_URL if use_local else WRAITH_REMOTE_URL
        endpoint = f"{api_url}/crawl"
        
        payload = {
            "url": url,
            "title": f"Status Page Check: {url}",
            "take_screenshot": False,
            "javascript_enabled": True,
            "ocr_extraction": False,
            "markdown_extraction": "enhanced",
            "response_format": "minimal"
        }
        
        response = requests.post(endpoint, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Wraith API error for {url}: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error crawling {url}: {str(e)}")
        return None

def analyze_status_page_content(content: str, url: str) -> Dict[str, Any]:
    """Analyze content to determine if it's a status page and identify the provider."""
    content_lower = content.lower()
    
    # Check if it's likely a status page
    status_indicators = [
        "operational", "degraded", "outage", "incident", "uptime",
        "system status", "service status", "all systems", "components",
        "maintenance", "disruption", "availability"
    ]
    
    status_score = sum(1 for indicator in status_indicators if indicator in content_lower)
    is_status_page = status_score >= 3
    
    # Identify provider
    detected_provider = "unknown"
    provider_confidence = 0
    
    for provider, info in STATUS_PAGE_PROVIDERS.items():
        if provider == "custom":
            continue
            
        matches = 0
        for indicator in info["indicators"]:
            if indicator.lower() in content_lower:
                matches += 1
        
        if info["pattern"] and re.search(info["pattern"], content_lower):
            matches += 2
        
        if matches > provider_confidence:
            provider_confidence = matches
            detected_provider = provider
    
    if detected_provider == "unknown" and is_status_page:
        detected_provider = "custom"
    
    # Extract status information
    status_info = {
        "all_operational": any(phrase in content_lower for phrase in ["all systems operational", "all services operational"]),
        "has_incidents": "incident" in content_lower,
        "has_maintenance": "maintenance" in content_lower
    }
    
    return {
        "is_status_page": is_status_page,
        "confidence_score": status_score,
        "provider": detected_provider,
        "provider_confidence": provider_confidence,
        "status_info": status_info,
        "url": url
    }

@mcp.tool()
async def find_status_pages(
    services: List[str],
    check_awesome_list: bool = True,
    use_local_wraith: bool = False,
    max_attempts_per_service: int = 5,
    ctx: Context = None
) -> Dict[str, Any]:
    '''
    Find and analyze status pages for given services.
    
    Args:
        services: List of service names or domains to find status pages for
        check_awesome_list: Whether to check the awesome-status-pages repository
        use_local_wraith: Whether to use local Wraith server instead of remote
        max_attempts_per_service: Maximum URLs to try per service
        ctx: Context object for logging and progress
        
    Returns:
        Dictionary with found status pages and analysis
    '''
    results = {
        "services": {},
        "summary": {
            "total_services": len(services),
            "found_status_pages": 0,
            "providers": {}
        }
    }
    
    # First, get known status pages from awesome-status-pages if requested
    known_status_pages = {}
    if check_awesome_list:
        try:
            # Try to get the awesome list content
            awesome_url = "https://raw.githubusercontent.com/ivbeg/awesome-status-pages/master/README.md"
            response = requests.get(awesome_url, timeout=10)
            if response.status_code == 200:
                content = response.text
                # Extract public status pages section
                public_section = re.search(r"## Public status pages(.*?)(?=##|\Z)", content, re.DOTALL)
                if public_section:
                    # Extract service names and URLs
                    matches = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", public_section.group(1))
                    for name, url in matches:
                        known_status_pages[name.lower()] = url
                        logger.info(f"Found known status page: {name} -> {url}")
        except Exception as e:
            logger.error(f"Error fetching awesome-status-pages: {str(e)}")
    
    # Process each service
    for service in services:
        service_lower = service.lower().strip()
        results["services"][service] = {
            "service": service,
            "status_page_found": False,
            "attempts": [],
            "final_result": None
        }
        
        # Check if we have a known status page
        if service_lower in known_status_pages:
            url = known_status_pages[service_lower]
            logger.info(f"Using known status page for {service}: {url}")
            
            # Crawl and analyze
            crawl_result = await crawl_with_wraith(url, use_local_wraith)
            if crawl_result and crawl_result.get("success"):
                content = crawl_result.get("markdown", "")
                analysis = analyze_status_page_content(content, url)
                
                results["services"][service]["status_page_found"] = analysis["is_status_page"]
                results["services"][service]["final_result"] = analysis
                results["services"][service]["attempts"].append({
                    "url": url,
                    "source": "awesome-list",
                    "success": True,
                    "analysis": analysis
                })
                
                if analysis["is_status_page"]:
                    results["summary"]["found_status_pages"] += 1
                    provider = analysis["provider"]
                    results["summary"]["providers"][provider] = results["summary"]["providers"].get(provider, 0) + 1
                
                continue
        
        # Generate and test URLs
        potential_urls = guess_status_page_urls(service)
        found = False
        
        for i, url in enumerate(potential_urls[:max_attempts_per_service]):
            logger.info(f"Trying URL for {service}: {url}")
            
            # Crawl the URL
            crawl_result = await crawl_with_wraith(url, use_local_wraith)
            
            if crawl_result and crawl_result.get("success"):
                content = crawl_result.get("markdown", "")
                analysis = analyze_status_page_content(content, url)
                
                results["services"][service]["attempts"].append({
                    "url": url,
                    "source": "guessed",
                    "success": True,
                    "analysis": analysis
                })
                
                if analysis["is_status_page"] and analysis["confidence_score"] >= 3:
                    results["services"][service]["status_page_found"] = True
                    results["services"][service]["final_result"] = analysis
                    results["summary"]["found_status_pages"] += 1
                    
                    provider = analysis["provider"]
                    results["summary"]["providers"][provider] = results["summary"]["providers"].get(provider, 0) + 1
                    
                    found = True
                    break
            else:
                results["services"][service]["attempts"].append({
                    "url": url,
                    "source": "guessed",
                    "success": False,
                    "error": "Failed to crawl"
                })
            
            # Small delay between attempts
            await asyncio.sleep(0.5)
        
        if not found:
            # Pick the best attempt if any had partial matches
            best_attempt = None
            best_score = 0
            
            for attempt in results["services"][service]["attempts"]:
                if attempt.get("success") and attempt.get("analysis"):
                    score = attempt["analysis"]["confidence_score"]
                    if score > best_score:
                        best_score = score
                        best_attempt = attempt
            
            if best_attempt and best_score > 0:
                results["services"][service]["final_result"] = best_attempt["analysis"]
    
    return results

@mcp.tool()
async def analyze_status_page(
    url: str,
    use_local_wraith: bool = False,
    ctx: Context = None
) -> Dict[str, Any]:
    '''
    Analyze a specific status page URL.
    
    Args:
        url: URL of the status page to analyze
        use_local_wraith: Whether to use local Wraith server
        ctx: Context object for logging and progress
        
    Returns:
        Dictionary with detailed status page analysis
    '''
    result = {
        "url": url,
        "success": False,
        "analysis": None,
        "error": None
    }
    
    try:
        # Crawl the page
        crawl_result = await crawl_with_wraith(url, use_local_wraith)
        
        if crawl_result and crawl_result.get("success"):
            content = crawl_result.get("markdown", "")
            analysis = analyze_status_page_content(content, url)
            
            # Extract additional details
            content_lower = content.lower()
            
            # Find components/services
            components = []
            component_patterns = [
                r"(?:component|service):\s*([^\n]+)",
                r"(?:✓|✔|🟢)\s*([^\n]+)",
                r"(?:operational|online|up)\s*-\s*([^\n]+)"
            ]
            
            for pattern in component_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                components.extend(matches)
            
            # Find recent incidents
            incidents = []
            incident_patterns = [
                r"incident:\s*([^\n]+)",
                r"(?:issue|problem|outage):\s*([^\n]+)"
            ]
            
            for pattern in incident_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                incidents.extend(matches[:5])  # Limit to 5 most recent
            
            analysis["components"] = list(set(components))[:20]  # Unique, limited to 20
            analysis["recent_incidents"] = incidents
            analysis["page_title"] = crawl_result.get("title", "")
            
            result["success"] = True
            result["analysis"] = analysis
        else:
            result["error"] = "Failed to crawl the page"
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error analyzing {url}: {str(e)}")
    
    return result

@mcp.tool()
async def get_awesome_status_pages(
    category: str = "all",
    ctx: Context = None
) -> Dict[str, Any]:
    '''
    Get status pages from the awesome-status-pages repository.
    
    Args:
        category: Category to filter ("all", "opensource", "services", "public")
        ctx: Context object for logging and progress
        
    Returns:
        Dictionary with categorized status pages
    '''
    result = {
        "success": False,
        "categories": {},
        "total_count": 0,
        "error": None
    }
    
    try:
        # Fetch the awesome list
        url = "https://raw.githubusercontent.com/ivbeg/awesome-status-pages/master/README.md"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            content = response.text
            
            # Parse different sections
            sections = {
                "opensource": r"## Opensource\s*(.*?)(?=##|\Z)",
                "services": r"## Services\s*(.*?)(?=##|\Z)",
                "public": r"## Public status pages\s*(.*?)(?=##|\Z)"
            }
            
            for section_name, pattern in sections.items():
                if category != "all" and category != section_name:
                    continue
                
                section_match = re.search(pattern, content, re.DOTALL)
                if section_match:
                    section_content = section_match.group(1)
                    
                    # Extract items with URLs
                    items = []
                    matches = re.findall(r"\*\s*\[([^\]]+)\]\(([^)]+)\)\s*-\s*([^\n]+)", section_content)
                    
                    for name, url, description in matches:
                        items.append({
                            "name": name.strip(),
                            "url": url.strip(),
                            "description": description.strip()
                        })
                    
                    result["categories"][section_name] = items
                    result["total_count"] += len(items)
            
            result["success"] = True
        else:
            result["error"] = f"Failed to fetch awesome list: HTTP {response.status_code}"
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error fetching awesome list: {str(e)}")
    
    return result

# Start the MCP server
if __name__ == "__main__":
    try:
        logger.info(f"Starting Status Page Finder v{__version__}")
        mcp.run(transport='stdio')
    except Exception as e:
        logger.critical(f"Failed to start: {str(e)}", exc_info=True)
        sys.exit(1)
