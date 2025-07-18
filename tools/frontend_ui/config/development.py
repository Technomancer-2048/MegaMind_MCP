"""Development configuration for frontend UI service"""

DEBUG = True
AUTO_APPROVE = False  # Manual approval required in development
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "dev"
DB_PASSWORD = ""
DATABASE = "megamind_database"
LOG_LEVEL = "DEBUG"

# Frontend UI specific settings
REFRESH_INTERVAL = 30  # seconds
MAX_CHUNKS_PER_PAGE = 50
ENABLE_SEARCH_HIGHLIGHTING = True
ENABLE_CONTEXT_PREVIEW = True

# Security settings (relaxed for development)
CORS_ENABLED = True
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5004"]
RATE_LIMITING = False