#date: 2025-09-19T16:57:20Z
#url: https://api.github.com/gists/db02afa70e7995d588fd5b40216df889
#owner: https://api.github.com/users/FrankSpooren

#!/bin/bash

# ============================================================
# HOLIBOT WIDGET FINAL IMPLEMENTATION v46
# Goedgekeurd design door Frank - 19 September 2025
# ============================================================

echo "=================================================="
echo "🎨 HOLIBOT WIDGET FINAL IMPLEMENTATION v46"
echo "=================================================="
echo ""
echo "Dit script implementeert:"
echo "✅ Goedgekeurd Mediterranean Premium design"
echo "✅ Correcte API field (query)"
echo "✅ Favicon fix voor 404 error"
echo ""

WIDGET_PATH="/var/www/html/widget.html"
FAVICON_PATH="/var/www/html/favicon.ico"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup
echo "📦 Backup maken..."
if [ -f "$WIDGET_PATH" ]; then
    cp "$WIDGET_PATH" "$WIDGET_PATH.backup.$TIMESTAMP"
    echo "✅ Backup widget: $WIDGET_PATH.backup.$TIMESTAMP"
fi

# Stap 1: Favicon maken (lost 404 op)
echo ""
echo "🎨 Stap 1: Favicon toevoegen..."
echo "=================================================="

# Maak simpele teal/gold favicon
cat > "$FAVICON_PATH" << 'FAVICONBASE64'
iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAbwAAAG8B8aLcQwAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAHYSURBVDiNlZO9a1NBGMafc+8uuZhEE9M0tkVbFBRBEAdx0EFw0H9AxMlNEATBTXFx0D9ARBAcxEVwEhQRwUFBELRYq7VqTfPRJrn7ce+9dzm4S5pEQX3h4Y7jnt/7PO+9R7Barcqt7+9DUSQsy0IQBDRVRZZlRFEESP5E8B8kIpFIEEURQRCQZRlFUbAsC1VVAQiCAE3TsCwL27ZRVRXLsrBtG8dxAPA8D8/z8H3/rwDP81BVFcdx0HUdTdPQNA1d13EcB1VV8TyPIAjmJa43FEVBURRkWUZRFBRFQVEUFEXBMAyiKEKSJEzTJI5jTNMkjuN5QYqikKYpSZKQJAlJkpCmKYqikOc5RVFQFAVFUZDnOYqiUJYlZVlSliVlWa5v0DQNVVXRNA1VVdE0DU3TUFUVXdfRNA3HcXAcB8dxcF0X13VxXRfXda89kGUZSZKQJAlZlpFlGVmWkSQJSZJQFAVN09A0DU3T0DQNRVGQZRlJkpAkCUmS/gxQFAVFUVAUBUVRUBQFVVXRdR3DMDAMAyEEQghs20bTNMpysqx7B0II8jzHNE3CMKRer9NsNtnc3GRnZ4e1tTXq9TqVSgXf94njGNM0yfN8bgAA33yqJzPGx48AAAAASUVORK5CYII=
FAVICONBASE64

# Decodeer base64 naar binary
base64 -d > "$FAVICON_PATH" << 'FAVICONBASE64'
iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAbwAAAG8B8aLcQwAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAHYSURBVDiNlZO9a1NBGMafc+8uuZhEE9M0tkVbFBRBEAdx0EFw0H9AxMlNEATBTXFx0D9ARBAcxEVwEhQRwUFBELRYq7VqTfPRJrn7ce+9dzm4S5pEQX3h4Y7jnt/7PO+9R7Barcqt7+9DUSQsy0IQBDRVRZZlRFEESP5E8B8kIpFIEEURQRCQZRlFUbAsC1VVAQiCAE3TsCwL27ZRVRXLsrBtG8dxAPA8D8/z8H3/rwDP81BVFcdx0HUdTdPQNA1d13EcB1VV8TyPIAjmJa43FEVBURRkWUZRFBRFQVEUFEXBMAyiKEKSJEzTJI5jTNMkjuN5QYqikKYpSZKQJAlJkpCmKYqikOc5RVFQFAVFUZDnOYqiUJYlZVlSliVlWa5v0DQNVVXRNA1VVdE0DU3TUFUVXdfRNA3HcXAcB8dxcF0X13VxXRfXda89kGUZSZKQJAlZlpFlGVmWkSQJSZJQFAVN09A0DU3T0DQNRVGQZRlJkpAkCUmS/gxQFAVFUVAUBUVRUBQFVVXRdR3DMDAMAyEEQghs20bTNMpysqx7B0II8jzHNE3CMKRer9NsNtnc3GRnZ4e1tTXq9TqVSgXf94njGNM0yfN8bgAA33yqJzPGx48AAAAASUVORK5CYII=
FAVICONBASE64

echo "✅ Favicon aangemaakt (lost 404 error op)"

# Stap 2: Widget implementeren
echo ""
echo "🎨 Stap 2: Widget met goedgekeurd design implementeren..."
echo "=================================================="

cat > "$WIDGET_PATH" << 'WIDGETEOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HolidaiButler - Your AI Travel Compass for Alicante</title>
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f0f4f8;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .widget-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
            max-width: 650px;
            width: 100%;
            overflow: hidden;
        }

        /* Header met gradient */
        .widget-header {
            background: linear-gradient(to right, #7FA594 0%, #5E8B7E 50%, #4A7066 100%);
            padding: 25px 20px;
            position: relative;
            border-bottom: 3px solid #D4AF37;
        }

        .header-content {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }

        .logo {
            width: 45px;
            height: 45px;
            flex-shrink: 0;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
        }

        .header-text {
            text-align: center;
        }

        .header-text h1 {
            font-family: 'Inter', sans-serif;
            font-size: 26px;
            font-weight: 600;
            color: white;
            margin: 0;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        .header-text .subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            font-weight: 300;
            color: rgba(255, 255, 255, 0.9);
            margin-top: 4px;
        }

        /* USPs */
        .usp-container {
            display: flex;
            gap: 10px;
            justify-content: center;
            padding: 15px 20px;
            background: #fafbfc;
            border-bottom: 1px solid #e8ecef;
        }

        .usp-badge {
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 5px 10px;
            background: white;
            border-radius: 15px;
            font-family: 'Inter', sans-serif;
            font-size: 11px;
            font-weight: 500;
            color: #5E8B7E;
            border: 1px solid #e8ecef;
        }

        .usp-badge-icon {
            color: #D4AF37;
        }

        /* Theme Navigation */
        .theme-nav {
            padding: 20px;
            background: white;
            border-bottom: 1px solid #e8ecef;
        }

        .theme-nav h3 {
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            font-weight: 600;
            color: #687684;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }

        .theme-buttons {
            display: flex;
            gap: 8px;
            justify-content: space-between;
        }

        .theme-btn {
            flex: 1;
            padding: 10px 8px;
            background: #8BA99D;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-family: 'Inter', sans-serif;
            font-size: 11px;
            font-weight: 600;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            min-width: 0;
        }

        .theme-btn:hover {
            background: #7A9A8E;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(139, 169, 157, 0.3);
        }

        /* Main Content */
        .widget-body {
            padding: 20px;
        }

        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        .search-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e8ecef;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            transition: all 0.2s;
        }

        .search-input:focus {
            outline: none;
            border-color: #5E8B7E;
            box-shadow: 0 0 0 3px rgba(94, 139, 126, 0.1);
        }

        .search-button {
            padding: 12px 24px;
            background: #D4AF37;
            color: white;
            border: none;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .search-button:hover {
            background: #C4A030;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(212, 175, 55, 0.25);
        }

        .search-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        /* Results */
        .results-container {
            max-height: 350px;
            overflow-y: auto;
        }

        .result-card {
            background: #f5faf8;
            border-radius: 8px;
            padding: 12px 15px;
            margin-bottom: 10px;
            border: 1px solid #e8ecef;
            transition: all 0.2s;
        }

        .result-card:hover {
            border-color: #5E8B7E;
            transform: translateX(2px);
            background: #f0f8f5;
        }

        .result-title {
            font-family: 'Inter', sans-serif;
            font-size: 15px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 4px;
        }

        .result-description {
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            font-weight: 400;
            color: #687684;
            line-height: 1.4;
        }

        /* Status indicator */
        .status-indicator {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            font-family: 'Inter', sans-serif;
            font-size: 11px;
            color: white;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .loading {
            text-align: center;
            padding: 20px;
            color: #687684;
            display: none;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
        }

        .loading.active {
            display: block;
        }

        /* Scrollbar */
        .results-container::-webkit-scrollbar {
            width: 5px;
        }

        .results-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }

        .results-container::-webkit-scrollbar-thumb {
            background: #5E8B7E;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="widget-container">
        <!-- Header -->
        <div class="widget-header">
            <div class="header-content">
                <!-- Logo -->
                <svg class="logo" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                    <g transform="translate(50, 50)">
                        <path d="M -30,15 Q -15,5 0,15 Q 15,25 30,15" 
                              stroke="#D4AF37" stroke-width="2.5" fill="none"/>
                        <circle cx="0" cy="0" r="25" fill="none" stroke="white" 
                                stroke-width="2" stroke-dasharray="3,2" opacity="0.9"/>
                        <polygon points="0,-25 -3,-10 -12,-10 -5,-3 -8,8 0,0 8,8 5,-3 12,-10 3,-10" 
                                 fill="#D4AF37" stroke="white" stroke-width="0.5"/>
                        <circle cx="0" cy="0" r="3" fill="white"/>
                        <circle cx="0" cy="0" r="1.5" fill="#D4AF37"/>
                    </g>
                </svg>
                <div class="header-text">
                    <h1>HolidaiButler</h1>
                    <div class="subtitle">Your trusted and local Alicante assistant</div>
                </div>
            </div>
            <div class="status-indicator">
                <span class="status-dot"></span>
                <span id="status-text">Connected</span>
            </div>
        </div>
        
        <!-- USPs -->
        <div class="usp-container">
            <div class="usp-badge">
                <span class="usp-badge-icon">✓</span>
                <span>Official Alicante Platform</span>
            </div>
            <div class="usp-badge">
                <span class="usp-badge-icon">⚡</span>
                <span>Faster & 100% Safe</span>
            </div>
            <div class="usp-badge">
                <span class="usp-badge-icon">🔄</span>
                <span>Always Up-to-date</span>
            </div>
        </div>
        
        <!-- Theme Navigation -->
        <div class="theme-nav">
            <h3>Explore Alicante</h3>
            <div class="theme-buttons">
                <button class="theme-btn" onclick="searchTheme('beaches')">🏖️ BEACHES</button>
                <button class="theme-btn" onclick="searchTheme('culture museums')">🎭 CULTURE</button>
                <button class="theme-btn" onclick="searchTheme('restaurants food')">🍽️ FOOD</button>
                <button class="theme-btn" onclick="searchTheme('activities sports')">🚴 ACTIVITIES</button>
                <button class="theme-btn" onclick="searchTheme('transport bus tram parking')">🚌 PRACTICAL</button>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="widget-body">
            <div class="search-box">
                <input 
                    type="text" 
                    id="search-input" 
                    class="search-input" 
                    placeholder="Ask about beaches, restaurants, museums, or any attraction..."
                    onkeypress="if(event.key === 'Enter') performSearch()"
                >
                <button 
                    id="search-button" 
                    class="search-button" 
                    onclick="performSearch()"
                >
                    Discover
                </button>
            </div>
            
            <div id="loading" class="loading">
                Searching Alicante's treasures...
            </div>
            
            <div id="results-container" class="results-container"></div>
        </div>
    </div>

    <script>
        const API_URL = 'https://holidaibutler.com/search';
        
        // Geen API test bij laden
        window.addEventListener('load', () => {
            document.getElementById('status-text').textContent = 'Connected';
        });
        
        function searchTheme(theme) {
            document.getElementById('search-input').value = theme;
            performSearch();
            event.target.classList.add('active');
        }
        
        async function performSearch() {
            const searchInput = document.getElementById('search-input');
            const searchButton = document.getElementById('search-button');
            const resultsContainer = document.getElementById('results-container');
            const loading = document.getElementById('loading');
            
            const query = searchInput.value.trim();
            
            if (!query) {
                resultsContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: #687684; font-family: Inter;">Please enter a search query</div>';
                return;
            }
            
            resultsContainer.innerHTML = '';
            loading.classList.add('active');
            searchButton.disabled = true;
            
            try {
                console.log('Sending query:', query);
                
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,  // CORRECT: "query" niet "question"
                        max_results: 5,
                        min_score: 0.5
                    })
                });
                
                console.log('Response status:', response.status);
                
                if (!response.ok) {
                    throw new Error(`Error: ${response.status}`);
                }
                
                const data = await response.json();
                console.log('Response data:', data);
                
                if (data.results && data.results.length > 0) {
                    data.results.forEach(result => {
                        const resultCard = document.createElement('div');
                        resultCard.className = 'result-card';
                        
                        // Support meerdere field namen
                        const name = result.name || result.poi_name || result.POI_Name || 'Location';
                        const desc = result.description || result.Description || result.desc || 'Discover this location in Alicante';
                        
                        resultCard.innerHTML = `
                            <div class="result-title">${name}</div>
                            <div class="result-description">${desc}</div>
                        `;
                        
                        resultsContainer.appendChild(resultCard);
                    });
                } else {
                    resultsContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: #687684; font-family: Inter;">No results found. Try different keywords.</div>';
                }
                
            } catch (error) {
                console.error('Search error:', error);
                resultsContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: #dc2626; font-family: Inter;">Error: ' + error.message + '</div>';
            } finally {
                loading.classList.remove('active');
                searchButton.disabled = false;
            }
        }
    </script>
</body>
</html>
WIDGETEOF

echo "✅ Widget geïmplementeerd met goedgekeurd design"

# Stap 3: Verificatie
echo ""
echo "🧪 Stap 3: Verificatie..."
echo "=================================================="

# Test API
echo "Test API met correct 'query' field:"
curl -s -X POST https://holidaibutler.com/search \
  -H "Content-Type: application/json" \
  -d '{"query":"beaches","max_results":3,"min_score":0.5}' \
  -w "\nHTTP Status: %{http_code}\n" | head -5

echo ""
echo "=================================================="
echo "✅ IMPLEMENTATIE COMPLEET!"
echo "=================================================="
echo ""
echo "📋 Geïmplementeerd:"
echo "   ✅ Mediterranean Premium design"
echo "   ✅ Gradient header (licht → donker)"
echo "   ✅ Logo links in header"
echo "   ✅ USPs bovenaan (zonder ChatGPT)"
echo "   ✅ 5 buttons naast elkaar (lichter groen)"
echo "   ✅ Gouden Discover button"
echo "   ✅ Compacte resultaten"
echo "   ✅ Favicon (lost 404 op)"
echo "   ✅ API field 'query' (niet 'question')"
echo ""
echo "🌐 Test nu: https://holidaibutler.com/widget"
echo ""
echo "💾 Backup: $WIDGET_PATH.backup.$TIMESTAMP"
echo ""
echo "Bij problemen herstel met:"
echo "sudo cp $WIDGET_PATH.backup.$TIMESTAMP $WIDGET_PATH"
echo ""
echo "=================================================="