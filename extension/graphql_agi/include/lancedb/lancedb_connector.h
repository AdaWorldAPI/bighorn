#pragma once

/**
 * LanceDB Connector
 *
 * Provides integration with LanceDB for vector storage and search.
 * Features:
 * - Native LanceDB protocol support
 * - Automatic embedding generation
 * - Hybrid search (vector + filters)
 * - Efficient batch operations
 * - Index management (IVF-PQ, HNSW)
 */

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <unordered_map>

namespace kuzu {
namespace graphql_agi {

/**
 * Vector data types
 */
using Vector = std::vector<float>;
using VectorBatch = std::vector<Vector>;

/**
 * LanceDB table schema definition
 */
struct LanceDBColumn {
    std::string name;
    std::string type;  // "vector", "string", "int64", "float64", "bool", "timestamp"
    std::optional<uint32_t> dimension;  // For vector columns
    bool nullable = true;
};

struct LanceDBTableSchema {
    std::string tableName;
    std::vector<LanceDBColumn> columns;
    std::string vectorColumn;
    uint32_t vectorDimension = 1536;
};

/**
 * LanceDB index configuration
 */
struct LanceDBIndexConfig {
    enum class IndexType {
        None,
        IVF_PQ,      // Inverted File with Product Quantization
        IVF_FLAT,    // Inverted File with flat vectors
        IVF_HNSW_SQ, // IVF with HNSW sub-index and scalar quantization
        HNSW         // Hierarchical Navigable Small World
    };

    IndexType type = IndexType::IVF_PQ;

    // IVF parameters
    uint32_t numPartitions = 256;
    uint32_t numSubVectors = 96;

    // HNSW parameters
    uint32_t m = 16;
    uint32_t efConstruction = 200;

    // General parameters
    std::string distanceMetric = "cosine";  // "cosine", "l2", "dot"
    bool accelerateIndex = true;
};

/**
 * Search result from LanceDB
 */
struct LanceDBSearchResult {
    std::string id;
    float score;
    Vector vector;
    std::unordered_map<std::string, std::string> metadata;
};

/**
 * LanceDB connection configuration
 */
struct LanceDBConfig {
    std::string uri;              // "lance://path", "s3://bucket/path", etc.
    std::string region;           // For cloud storage
    std::optional<std::string> apiKey;
    std::optional<std::string> storageOptions;

    // Connection pool settings
    uint32_t maxConnections = 10;
    uint32_t connectionTimeoutMs = 5000;

    // Batch settings
    uint32_t batchSize = 1000;
    bool asyncWrites = true;
};

/**
 * LanceDB Connector class
 */
class LanceDBConnector {
public:
    LanceDBConnector(const LanceDBConfig& config);
    ~LanceDBConnector();

    /**
     * Connection management
     */
    bool connect();
    void disconnect();
    bool isConnected() const;

    /**
     * Table operations
     */
    bool createTable(const LanceDBTableSchema& schema);
    bool dropTable(const std::string& tableName);
    bool tableExists(const std::string& tableName);
    std::vector<std::string> listTables();
    std::optional<LanceDBTableSchema> getTableSchema(const std::string& tableName);

    /**
     * Index operations
     */
    bool createIndex(const std::string& tableName, const LanceDBIndexConfig& config);
    bool dropIndex(const std::string& tableName);
    bool hasIndex(const std::string& tableName);

    /**
     * Data operations
     */
    struct InsertResult {
        bool success;
        uint64_t rowsInserted;
        std::optional<std::string> error;
    };

    InsertResult insert(const std::string& tableName,
                         const std::vector<std::unordered_map<std::string, std::string>>& rows,
                         const VectorBatch& vectors);

    InsertResult upsert(const std::string& tableName,
                         const std::string& keyColumn,
                         const std::vector<std::unordered_map<std::string, std::string>>& rows,
                         const VectorBatch& vectors);

    bool deleteById(const std::string& tableName, const std::vector<std::string>& ids);
    bool deleteByFilter(const std::string& tableName, const std::string& filter);

    /**
     * Search operations
     */
    struct SearchOptions {
        uint32_t topK = 10;
        std::optional<std::string> filter;
        std::optional<std::vector<std::string>> selectColumns;
        bool includeVectors = false;
        float refineFactor = 1.0;  // For re-ranking

        // Pre-filter vs post-filter
        bool preFilter = true;

        // Distance threshold
        std::optional<float> distanceThreshold;
    };

    std::vector<LanceDBSearchResult> search(const std::string& tableName,
                                              const Vector& queryVector,
                                              const SearchOptions& options = {});

    std::vector<LanceDBSearchResult> searchBatch(const std::string& tableName,
                                                   const VectorBatch& queryVectors,
                                                   const SearchOptions& options = {});

    /**
     * Hybrid search (vector + full-text)
     */
    struct HybridSearchOptions : SearchOptions {
        std::optional<std::string> textQuery;
        std::optional<std::vector<std::string>> textColumns;
        float vectorWeight = 0.7;
        float textWeight = 0.3;
    };

    std::vector<LanceDBSearchResult> hybridSearch(const std::string& tableName,
                                                    const Vector& queryVector,
                                                    const HybridSearchOptions& options);

    /**
     * Aggregation operations
     */
    uint64_t count(const std::string& tableName,
                    const std::optional<std::string>& filter = std::nullopt);

    /**
     * Statistics and info
     */
    struct TableStats {
        uint64_t rowCount;
        uint64_t sizeBytes;
        uint32_t numFragments;
        bool hasIndex;
        std::optional<std::string> indexType;
    };

    std::optional<TableStats> getTableStats(const std::string& tableName);

private:
    LanceDBConfig config_;
    bool connected_ = false;

    // Implementation details (would use LanceDB C++ SDK)
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Vector Store wrapper for Kuzu integration
 * Provides a high-level API for storing and retrieving vectors
 * associated with Kuzu graph entities
 */
class VectorStore {
public:
    VectorStore(std::shared_ptr<LanceDBConnector> connector);
    ~VectorStore() = default;

    /**
     * Store a vector with associated entity ID
     */
    bool storeVector(const std::string& entityId,
                      const std::string& entityType,
                      const Vector& vector,
                      const std::unordered_map<std::string, std::string>& metadata = {});

    /**
     * Store vectors in batch
     */
    bool storeVectorBatch(const std::vector<std::string>& entityIds,
                           const std::string& entityType,
                           const VectorBatch& vectors,
                           const std::vector<std::unordered_map<std::string, std::string>>& metadata = {});

    /**
     * Get vector by entity ID
     */
    std::optional<Vector> getVector(const std::string& entityId,
                                      const std::string& entityType);

    /**
     * Find similar entities
     */
    struct SimilarEntity {
        std::string entityId;
        std::string entityType;
        float similarity;
        std::unordered_map<std::string, std::string> metadata;
    };

    std::vector<SimilarEntity> findSimilar(const Vector& queryVector,
                                             const std::string& entityType,
                                             uint32_t topK = 10,
                                             float threshold = 0.0);

    std::vector<SimilarEntity> findSimilarByEntity(const std::string& entityId,
                                                      const std::string& entityType,
                                                      uint32_t topK = 10);

    /**
     * Delete vector
     */
    bool deleteVector(const std::string& entityId, const std::string& entityType);

    /**
     * Sync with Kuzu graph
     * Ensures vector store is consistent with graph database
     */
    struct SyncResult {
        uint64_t added;
        uint64_t updated;
        uint64_t deleted;
        std::vector<std::string> errors;
    };

    SyncResult syncWithGraph(main::ClientContext* context,
                              const std::string& tableName,
                              const std::string& idColumn,
                              const std::string& textColumn);

private:
    std::shared_ptr<LanceDBConnector> connector_;
    std::string getTableName(const std::string& entityType);
};

} // namespace graphql_agi
} // namespace kuzu
