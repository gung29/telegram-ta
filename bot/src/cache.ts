export class TTLCache<T> {
  private store = new Map<string, { value: T; expires: number }>();

  constructor(private ttlMs: number) {}

  private normalize(key: string | number): string {
    return typeof key === "number" ? key.toString() : key;
  }

  get(key: string | number): T | null {
    const record = this.store.get(this.normalize(key));
    if (!record) return null;
    if (record.expires < Date.now()) {
      this.store.delete(this.normalize(key));
      return null;
    }
    return record.value;
  }

  set(key: string | number, value: T): void {
    this.store.set(this.normalize(key), { value, expires: Date.now() + this.ttlMs });
  }

  delete(key: string | number): void {
    this.store.delete(this.normalize(key));
  }
}
