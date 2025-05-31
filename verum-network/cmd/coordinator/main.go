package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/spf13/viper"
	"go.uber.org/zap"

	"github.com/your-org/verum/verum-network/internal/coordinator"
	"github.com/your-org/verum/verum-network/internal/models"
	"github.com/your-org/verum/verum-network/internal/network"
	"github.com/your-org/verum/verum-network/internal/utils"
)

var (
	version   = "0.1.0"
	buildTime = "unknown"
	gitCommit = "unknown"
)

type Config struct {
	Server struct {
		Port            int           `mapstructure:"port"`
		ReadTimeout     time.Duration `mapstructure:"read_timeout"`
		WriteTimeout    time.Duration `mapstructure:"write_timeout"`
		ShutdownTimeout time.Duration `mapstructure:"shutdown_timeout"`
	} `mapstructure:"server"`

	Coordinator struct {
		MaxVehicles       int           `mapstructure:"max_vehicles"`
		OptimizationCycle time.Duration `mapstructure:"optimization_cycle"`
		StressThreshold   float64       `mapstructure:"stress_threshold"`
		ResponseTimeout   time.Duration `mapstructure:"response_timeout"`
		EnableMeshNetwork bool          `mapstructure:"enable_mesh_network"`
	} `mapstructure:"coordinator"`

	Network struct {
		EnableTLS         bool          `mapstructure:"enable_tls"`
		CertFile          string        `mapstructure:"cert_file"`
		KeyFile           string        `mapstructure:"key_file"`
		DiscoveryPort     int           `mapstructure:"discovery_port"`
		HeartbeatInterval time.Duration `mapstructure:"heartbeat_interval"`
	} `mapstructure:"network"`

	Logging struct {
		Level      string `mapstructure:"level"`
		OutputPath string `mapstructure:"output_path"`
		ErrorPath  string `mapstructure:"error_path"`
	} `mapstructure:"logging"`
}

func main() {
	var (
		configFile  = flag.String("config", "", "Path to configuration file")
		showVersion = flag.Bool("version", false, "Show version information")
		logLevel    = flag.String("log-level", "info", "Log level (debug, info, warn, error)")
	)
	flag.Parse()

	if *showVersion {
		fmt.Printf("Verum Network Coordinator\n")
		fmt.Printf("Version: %s\n", version)
		fmt.Printf("Build Time: %s\n", buildTime)
		fmt.Printf("Git Commit: %s\n", gitCommit)
		os.Exit(0)
	}

	// Load configuration
	config, err := loadConfig(*configFile)
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Setup logging
	logger, err := setupLogging(config.Logging, *logLevel)
	if err != nil {
		log.Fatalf("Failed to setup logging: %v", err)
	}
	defer logger.Sync()

	logger.Info("Starting Verum Network Coordinator",
		zap.String("version", version),
		zap.String("build_time", buildTime),
	)

	// Create application context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Initialize coordinator
	coord, err := coordinator.New(coordinator.Config{
		MaxVehicles:       config.Coordinator.MaxVehicles,
		OptimizationCycle: config.Coordinator.OptimizationCycle,
		StressThreshold:   config.Coordinator.StressThreshold,
		ResponseTimeout:   config.Coordinator.ResponseTimeout,
		EnableMeshNetwork: config.Coordinator.EnableMeshNetwork,
	}, logger)
	if err != nil {
		logger.Fatal("Failed to create coordinator", zap.Error(err))
	}

	// Initialize network server
	networkServer, err := network.NewServer(network.Config{
		Port:              config.Network.DiscoveryPort,
		EnableTLS:         config.Network.EnableTLS,
		CertFile:          config.Network.CertFile,
		KeyFile:           config.Network.KeyFile,
		HeartbeatInterval: config.Network.HeartbeatInterval,
	}, coord, logger)
	if err != nil {
		logger.Fatal("Failed to create network server", zap.Error(err))
	}

	// Setup HTTP server
	router := setupHTTPRoutes(coord, logger)
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", config.Server.Port),
		Handler:      router,
		ReadTimeout:  config.Server.ReadTimeout,
		WriteTimeout: config.Server.WriteTimeout,
	}

	// Start coordinator
	if err := coord.Start(ctx); err != nil {
		logger.Fatal("Failed to start coordinator", zap.Error(err))
	}

	// Start network server
	if err := networkServer.Start(ctx); err != nil {
		logger.Fatal("Failed to start network server", zap.Error(err))
	}

	// Start HTTP server
	go func() {
		logger.Info("Starting HTTP server", zap.Int("port", config.Server.Port))
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("HTTP server failed", zap.Error(err))
		}
	}()

	// Wait for shutdown signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down server...")

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), config.Server.ShutdownTimeout)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		logger.Error("Server forced to shutdown", zap.Error(err))
	}

	// Stop coordinator and network server
	coord.Stop()
	networkServer.Stop()

	logger.Info("Server exited")
}

func loadConfig(configFile string) (*Config, error) {
	viper.SetConfigName("config")
	viper.SetConfigType("toml")
	viper.AddConfigPath(".")
	viper.AddConfigPath("/etc/verum/")
	viper.AddConfigPath("$HOME/.verum/")

	if configFile != "" {
		viper.SetConfigFile(configFile)
	}

	// Set defaults
	viper.SetDefault("server.port", 8081)
	viper.SetDefault("server.read_timeout", "30s")
	viper.SetDefault("server.write_timeout", "30s")
	viper.SetDefault("server.shutdown_timeout", "10s")

	viper.SetDefault("coordinator.max_vehicles", 1000)
	viper.SetDefault("coordinator.optimization_cycle", "5s")
	viper.SetDefault("coordinator.stress_threshold", 0.8)
	viper.SetDefault("coordinator.response_timeout", "2s")
	viper.SetDefault("coordinator.enable_mesh_network", false)

	viper.SetDefault("network.enable_tls", false)
	viper.SetDefault("network.discovery_port", 8082)
	viper.SetDefault("network.heartbeat_interval", "30s")

	viper.SetDefault("logging.level", "info")
	viper.SetDefault("logging.output_path", "stdout")
	viper.SetDefault("logging.error_path", "stderr")

	// Read environment variables
	viper.AutomaticEnv()
	viper.SetEnvPrefix("VERUM")

	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, err
		}
	}

	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		return nil, err
	}

	return &config, nil
}

func setupLogging(config struct {
	Level      string `mapstructure:"level"`
	OutputPath string `mapstructure:"output_path"`
	ErrorPath  string `mapstructure:"error_path"`
}, logLevel string) (*zap.Logger, error) {

	// Use command line log level if provided
	if logLevel != "info" {
		config.Level = logLevel
	}

	cfg := zap.NewProductionConfig()
	cfg.Level = zap.NewAtomicLevelAt(utils.ParseLogLevel(config.Level))
	cfg.OutputPaths = []string{config.OutputPath}
	cfg.ErrorOutputPaths = []string{config.ErrorPath}

	return cfg.Build()
}

func setupHTTPRoutes(coord *coordinator.Coordinator, logger *zap.Logger) *gin.Engine {
	// Set gin mode
	gin.SetMode(gin.ReleaseMode)

	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	// CORS middleware
	router.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	// Health check
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"version":   version,
			"timestamp": time.Now().Unix(),
		})
	})

	// Version info
	router.GET("/version", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"version":    version,
			"build_time": buildTime,
			"git_commit": gitCommit,
		})
	})

	// API routes
	api := router.Group("/api/v1")
	{
		// Vehicle registration
		api.POST("/vehicles/register", func(c *gin.Context) {
			var req models.VehicleRegistration
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			vehicle, err := coord.RegisterVehicle(req)
			if err != nil {
				logger.Error("Failed to register vehicle", zap.Error(err))
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusCreated, vehicle)
		})

		// Vehicle status update
		api.PUT("/vehicles/:id/status", func(c *gin.Context) {
			vehicleID := c.Param("id")

			var status models.VehicleStatus
			if err := c.ShouldBindJSON(&status); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			if err := coord.UpdateVehicleStatus(vehicleID, status); err != nil {
				logger.Error("Failed to update vehicle status",
					zap.String("vehicle_id", vehicleID),
					zap.Error(err))
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"status": "updated"})
		})

		// Route optimization request
		api.POST("/routes/optimize", func(c *gin.Context) {
			var req models.RouteOptimizationRequest
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			routes, err := coord.OptimizeRoutes(req)
			if err != nil {
				logger.Error("Failed to optimize routes", zap.Error(err))
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, routes)
		})

		// Coordination statistics
		api.GET("/stats", func(c *gin.Context) {
			stats := coord.GetStatistics()
			c.JSON(http.StatusOK, stats)
		})

		// Active vehicles
		api.GET("/vehicles", func(c *gin.Context) {
			vehicles := coord.GetActiveVehicles()
			c.JSON(http.StatusOK, gin.H{
				"vehicles": vehicles,
				"count":    len(vehicles),
			})
		})

		// Traffic flow analysis
		api.GET("/traffic/flow", func(c *gin.Context) {
			flow := coord.GetTrafficFlow()
			c.JSON(http.StatusOK, flow)
		})
	}

	// WebSocket endpoint for real-time updates
	router.GET("/ws", func(c *gin.Context) {
		// WebSocket handler implementation would go here
		// This would provide real-time updates to connected clients
		c.JSON(http.StatusNotImplemented, gin.H{
			"error": "WebSocket endpoint not yet implemented",
		})
	})

	return router
}
