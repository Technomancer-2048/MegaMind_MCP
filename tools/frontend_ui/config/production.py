"""Production configuration for frontend UI service"""

DEBUG = False
AUTO_APPROVE = True  # Auto-approve enabled in production
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "gameapp"
DB_PASSWORD = ""
DATABASE = "megamind_database"
LOG_LEVEL = "INFO"

# Frontend UI specific settings
REFRESH_INTERVAL = 60  # seconds
MAX_CHUNKS_PER_PAGE = 20
ENABLE_SEARCH_HIGHLIGHTING = True
ENABLE_CONTEXT_PREVIEW = True

# Security settings (strict for production)
CORS_ENABLED = False
CORS_ORIGINS = []
RATE_LIMITING = True
RATE_LIMIT_REQUESTS = 100  # per minute