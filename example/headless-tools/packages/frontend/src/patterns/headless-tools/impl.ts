import {
  memoryPut as memoryPutDefinition,
  memoryGet as memoryGetDefinition,
  memoryList as memoryListDefinition,
  memorySearch as memorySearchDefinition,
  memoryForget as memoryForgetDefinition,
  geolocationGet as geolocationGetDefinition,
} from "./tools";

const DB_NAME = "agent-memory";
const DB_VERSION = 2;
const STORE_NAME = "memories";

interface Memory {
  key: string;
  value: unknown;
  tags: string[];
  createdAt: string;
  updatedAt: string;
  expiresAt?: string;
}

/** Strip reactive proxies so IndexedDB structured clone can persist the value. */
function toStoredValue<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function deleteDatabase(): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(DB_NAME);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(new Error("Failed to delete database"));
    request.onblocked = () => resolve();
  });
}

async function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => {
      reject(new Error("Failed to open memory database"));
    };

    request.onsuccess = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.close();
        deleteDatabase()
          .then(() => openDB())
          .then(resolve)
          .catch(reject);
        return;
      }
      resolve(db);
    };

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (db.objectStoreNames.contains(STORE_NAME)) {
        db.deleteObjectStore(STORE_NAME);
      }
      const store = db.createObjectStore(STORE_NAME, { keyPath: "key" });
      store.createIndex("tags", "tags", { multiEntry: true });
      store.createIndex("createdAt", "createdAt");
      store.createIndex("updatedAt", "updatedAt");
    };

    request.onblocked = () => {
      reject(new Error("Database blocked — please close other tabs using this app"));
    };
  });
}

async function withStore<T>(
  mode: IDBTransactionMode,
  callback: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    try {
      const transaction = db.transaction(STORE_NAME, mode);
      const store = transaction.objectStore(STORE_NAME);
      const request = callback(store);

      transaction.oncomplete = () => {
        db.close();
      };
      transaction.onerror = () => {
        db.close();
        reject(new Error("Memory operation failed"));
      };
      request.onsuccess = () => {
        resolve(request.result);
      };
      request.onerror = () => {
        reject(new Error("Memory operation failed"));
      };
    } catch (err) {
      db.close();
      reject(err);
    }
  });
}

async function getAllMemories(): Promise<Memory[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    try {
      const transaction = db.transaction(STORE_NAME, "readonly");
      const store = transaction.objectStore(STORE_NAME);
      const request = store.getAll();

      transaction.oncomplete = () => {
        db.close();
      };
      transaction.onerror = () => {
        db.close();
        reject(new Error("Failed to list memories"));
      };
      request.onsuccess = () => {
        const now = new Date().toISOString();
        const memories = (request.result as Memory[]).filter(
          (m) => !m.expiresAt || m.expiresAt > now,
        );
        resolve(memories);
      };
      request.onerror = () => {
        reject(new Error("Failed to list memories"));
      };
    } catch (err) {
      db.close();
      reject(err);
    }
  });
}

export const memoryPut = memoryPutDefinition.implement(
  async ({ key, value, tags = [], ttlDays }) => {
    const now = new Date();
    const memory: Memory = {
      key,
      value: toStoredValue(value),
      tags: toStoredValue(tags ?? []),
      createdAt: now.toISOString(),
      updatedAt: now.toISOString(),
      expiresAt: ttlDays
        ? new Date(now.getTime() + ttlDays * 24 * 60 * 60 * 1000).toISOString()
        : undefined,
    };

    const existing = await withStore("readonly", (store) => store.get(key));
    if (existing) memory.createdAt = existing.createdAt;

    await withStore("readwrite", (store) => store.put(memory));

    return {
      success: true,
      action: existing ? "updated" : "created",
      key,
      message: `Memory "${key}" ${existing ? "updated" : "saved"}${
        ttlDays ? ` (expires in ${ttlDays} days)` : ""
      }`,
    };
  },
);

export const memoryGet = memoryGetDefinition.implement(async ({ key }) => {
  const memory = await withStore<Memory | undefined>("readonly", (store) => store.get(key));

  if (!memory) {
    return { found: false, key, message: `No memory found with key "${key}"` };
  }

  if (memory.expiresAt && memory.expiresAt < new Date().toISOString()) {
    await withStore("readwrite", (store) => store.delete(key));
    return { found: false, key, message: `Memory "${key}" has expired` };
  }

  return {
    found: true,
    key,
    value: memory.value,
    tags: memory.tags,
    createdAt: memory.createdAt,
    updatedAt: memory.updatedAt,
    expiresAt: memory.expiresAt,
  };
});

export const memoryList = memoryListDefinition.implement(async ({ tags, limit = 20 }) => {
  let memories = await getAllMemories();

  if (tags && tags.length > 0) {
    memories = memories.filter((m) => tags.some((tag: string) => m.tags.includes(tag)));
  }

  memories.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
  memories = memories.slice(0, limit);

  return {
    count: memories.length,
    memories: memories.map((m) => ({
      key: m.key,
      value: m.value,
      tags: m.tags,
      updatedAt: m.updatedAt,
    })),
  };
});

export const memorySearch = memorySearchDefinition.implement(
  async ({ query, tags, limit = 10 }) => {
    let memories = await getAllMemories();

    if (tags && tags.length > 0) {
      memories = memories.filter((m) => tags.some((tag: string) => m.tags.includes(tag)));
    }

    const queryLower = query.toLowerCase();
    const matches = memories.filter((m) => {
      const keyMatch = m.key.toLowerCase().includes(queryLower);
      const valueStr = typeof m.value === "string" ? m.value : JSON.stringify(m.value);
      const valueMatch = valueStr.toLowerCase().includes(queryLower);
      const tagMatch = m.tags.some((t) => t.toLowerCase().includes(queryLower));
      return keyMatch || valueMatch || tagMatch;
    });

    matches.sort((a, b) => {
      const aExact = a.key.toLowerCase() === queryLower;
      const bExact = b.key.toLowerCase() === queryLower;
      if (aExact && !bExact) return -1;
      if (bExact && !aExact) return 1;
      return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
    });

    return {
      query,
      count: Math.min(matches.length, limit),
      total: matches.length,
      results: matches.slice(0, limit).map((m) => ({
        key: m.key,
        value: m.value,
        tags: m.tags,
        updatedAt: m.updatedAt,
      })),
    };
  },
);

export const memoryForget = memoryForgetDefinition.implement(
  async ({ key, tag, confirmForgetAll }) => {
    if (!key && !tag) {
      if (confirmForgetAll) {
        const db = await openDB();
        return new Promise((resolve, reject) => {
          const transaction = db.transaction(STORE_NAME, "readwrite");
          const store = transaction.objectStore(STORE_NAME);
          const request = store.clear();
          request.onsuccess = () => {
            db.close();
            resolve({
              success: true,
              action: "cleared_all",
              message: "All memories have been forgotten",
            });
          };
          request.onerror = () => {
            db.close();
            reject(new Error("Failed to clear memories"));
          };
        });
      }
      return {
        success: false,
        message: "Please specify a key or tag to forget, or set confirmForgetAll to true",
      };
    }

    if (key) {
      const existing = await withStore<Memory | undefined>("readonly", (store) => store.get(key));
      if (!existing) {
        return { success: false, key, message: `No memory found with key "${key}"` };
      }
      await withStore("readwrite", (store) => store.delete(key));
      return {
        success: true,
        action: "deleted",
        key,
        message: `Memory "${key}" has been forgotten`,
      };
    }

    if (tag) {
      const memories = await getAllMemories();
      const toDelete = memories.filter((m) => m.tags.includes(tag));
      for (const memory of toDelete) {
        await withStore("readwrite", (store) => store.delete(memory.key));
      }
      return {
        success: true,
        action: "deleted_by_tag",
        tag,
        count: toDelete.length,
        message: `Forgotten ${toDelete.length} memories with tag "${tag}"`,
      };
    }

    return { success: false, message: "Unexpected state" };
  },
);

export const geolocationGet = geolocationGetDefinition.implement(async ({ save = true }) => {
  if (!navigator.geolocation) {
    return { success: false, message: "Geolocation is not supported by this browser." };
  }

  const position = await new Promise<GeolocationPosition>((resolve, reject) => {
    navigator.geolocation.getCurrentPosition(resolve, reject, {
      enableHighAccuracy: true,
      timeout: 10_000,
      maximumAge: 5 * 60 * 1_000,
    });
  });

  const { latitude, longitude, accuracy } = position.coords;
  const timestamp = new Date(position.timestamp).toISOString();
  const locationData = { latitude, longitude, accuracy, timestamp };

  if (save) {
    const now = new Date();
    const memory: Memory = {
      key: "user_location",
      value: locationData,
      tags: ["location", "geolocation"],
      createdAt: now.toISOString(),
      updatedAt: now.toISOString(),
    };
    await withStore("readwrite", (store) => store.put(memory));
  }

  return {
    success: true,
    saved: save,
    latitude,
    longitude,
    accuracy,
    timestamp,
    message: `Location determined: ${latitude.toFixed(5)}, ${longitude.toFixed(5)} (±${Math.round(accuracy)} m)`,
  };
});
