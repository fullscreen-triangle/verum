module github.com/your-org/verum/verum-network

go 1.21

require (
	// gRPC and protobuf
	google.golang.org/grpc v1.59.0
	google.golang.org/protobuf v1.31.0
	
	// HTTP client/server
	github.com/gin-gonic/gin v1.9.1
	github.com/gorilla/websocket v1.5.1
	
	// Configuration
	github.com/spf13/viper v1.17.0
	github.com/spf13/cobra v1.8.0
	
	// Database and storage
	go.etcd.io/etcd/client/v3 v3.5.10
	github.com/redis/go-redis/v9 v9.3.0
	gorm.io/gorm v1.25.5
	gorm.io/driver/postgres v1.5.4
	
	// Networking and service discovery
	github.com/hashicorp/consul/api v1.25.1
	github.com/go-zookeeper/zk v1.0.3
	
	// Cryptography and security
	golang.org/x/crypto v0.15.0
	github.com/golang-jwt/jwt/v5 v5.1.0
	
	// Logging and monitoring
	go.uber.org/zap v1.26.0
	github.com/prometheus/client_golang v1.17.0
	github.com/opentracing/opentracing-go v1.2.0
	github.com/jaegertracing/jaeger-client-go v2.30.0+incompatible
	
	// Utilities and helpers
	github.com/google/uuid v1.4.0
	github.com/stretchr/testify v1.8.4
	golang.org/x/time v0.4.0
	golang.org/x/sync v0.5.0
	
	// Optimization algorithms
	github.com/golang/geo v0.0.0-20230421003525-6adc56603217
	gonum.org/v1/gonum v0.14.0
	
	// Message queues
	github.com/nats-io/nats.go v1.31.0
	github.com/segmentio/kafka-go v0.4.44
	
	// Testing and development
	github.com/golang/mock v1.6.0
	github.com/testcontainers/testcontainers-go v0.25.0
)

require (
	github.com/bytedance/sonic v1.9.1 // indirect
	github.com/cespare/xxhash/v2 v2.2.0 // indirect
	github.com/chenzhuoyu/base64x v0.0.0-20221115062448-fe3a3abad311 // indirect
	github.com/coreos/go-semver v0.3.0 // indirect
	github.com/coreos/go-systemd/v22 v22.3.2 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/fsnotify/fsnotify v1.7.0 // indirect
	github.com/gabriel-vasile/mimetype v1.4.2 // indirect
	github.com/gin-contrib/sse v0.1.0 // indirect
	github.com/go-playground/locales v0.14.1 // indirect
	github.com/go-playground/universal-translator v0.18.1 // indirect
	github.com/go-playground/validator/v10 v10.14.0 // indirect
	github.com/goccy/go-json v0.10.2 // indirect
	github.com/gogo/protobuf v1.3.2 // indirect
	github.com/hashicorp/go-cleanhttp v0.5.2 // indirect
	github.com/hashicorp/go-rootcerts v1.0.2 // indirect
	github.com/hashicorp/hcl v1.0.0 // indirect
	github.com/hashicorp/serf v0.10.1 // indirect
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20221227161230-091c0ba34f0a // indirect
	github.com/jackc/pgx/v5 v5.4.3 // indirect
	github.com/jinzhu/inflection v1.0.0 // indirect
	github.com/jinzhu/now v1.1.5 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/klauspost/cpuid/v2 v2.2.4 // indirect
	github.com/leodido/go-urn v1.2.4 // indirect
	github.com/magiconair/properties v1.8.7 // indirect
	github.com/mattn/go-isatty v0.0.19 // indirect
	github.com/mitchellh/mapstructure v1.5.0 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.2 // indirect
	github.com/nats-io/nkeys v0.4.6 // indirect
	github.com/nats-io/nuid v1.0.1 // indirect
	github.com/pelletier/go-toml/v2 v2.1.0 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/prometheus/client_model v0.5.0 // indirect
	github.com/prometheus/common v0.45.0 // indirect
	github.com/prometheus/procfs v0.12.0 // indirect
	github.com/spf13/afero v1.10.0 // indirect
	github.com/spf13/cast v1.5.0 // indirect
	github.com/spf13/jwalterweatherman v1.1.0 // indirect
	github.com/spf13/pflag v1.0.5 // indirect
	github.com/subosito/gotenv v1.6.0 // indirect
	github.com/twitchyliquid64/golang-asm v0.15.1 // indirect
	github.com/uber/jaeger-client-go v2.30.0+incompatible // indirect
	github.com/uber/jaeger-lib v2.4.1+incompatible // indirect
	github.com/ugorji/go/codec v1.2.11 // indirect
	go.etcd.io/etcd/api/v3 v3.5.10 // indirect
	go.etcd.io/etcd/client/pkg/v3 v3.5.10 // indirect
	go.uber.org/atomic v1.9.0 // indirect
	go.uber.org/multierr v1.9.0 // indirect
	golang.org/x/arch v0.3.0 // indirect
	golang.org/x/net v0.18.0 // indirect
	golang.org/x/sys v0.14.0 // indirect
	golang.org/x/text v0.14.0 // indirect
	google.golang.org/genproto/googleapis/api v0.0.0-20230711160842-782d3b101e98 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20230711160842-782d3b101e98 // indirect
	gopkg.in/ini.v1 v1.67.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
) 