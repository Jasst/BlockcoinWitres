// static/js/mining-worker.js
class MiningWorker {
    constructor() {
        this.running = false;
        this.encoder = new TextEncoder();
        self.onmessage = (e) => this.handleMessage(e);
    }

    handleMessage(e) {
        const { command, payload } = e.data;
        if (command === 'start') {
            this.startMining(payload);
        } else if (command === 'stop') {
            this.running = false; // Мягкая остановка
        }
    }

    async startMining({ last_proof, last_index, challenge, difficulty, maxIter, startTime }) {
        this.running = true;
        const target = '0'.repeat(difficulty);
        const BATCH_SIZE = 500; // Хешируем по 500 штук за раз (ускоряет crypto.subtle)

        let proof = 0;

        while (proof < maxIter && this.running) {
            const batchPromises = [];
            const batchProofs = [];

            // Формируем батч
            for (let i = 0; i < BATCH_SIZE && (proof + i) < maxIter; i++) {
                const currentProof = proof + i;
                const message = `${last_proof}${challenge}${currentProof}`;
                batchProofs.push(currentProof);
                batchPromises.push(this.sha256(message));
            }

            // Выполняем хеширование батча одновременно
            const hashes = await Promise.all(batchPromises);

            // Проверяем результаты
            for (let i = 0; i < hashes.length; i++) {
                if (hashes[i].startsWith(target)) {
                    this.running = false;
                    self.postMessage({
                        type: 'found',
                        found: true,
                        proof: batchProofs[i],
                        last_index, // Возвращаем last_index обратно!
                        elapsed: Date.now() - startTime
                    });
                    return;
                }
            }

            proof += batchPromises.length;

            // Отправляем прогресс и делаем паузу, чтобы воркер мог "дышать" и слушать команды
            if (proof % 50000 < BATCH_SIZE) {
                const elapsedSec = (Date.now() - startTime) / 1000;
                const hashrate = proof / elapsedSec;
                const remaining = maxIter - proof;
                const etaSec = hashrate > 0 ? remaining / hashrate : Infinity;

                self.postMessage({
                    type: 'progress',
                    progress: proof,
                    maxIter,
                    hashrate: Math.floor(hashrate),
                    eta: etaSec
                });

                // Пауза для обработки входящих сообщений (например, 'stop')
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }

        // Если прошли все итерации и ничего не нашли
        if (this.running) {
            self.postMessage({ type: 'not_found', found: false });
        }
    }

    async sha256(message) {
        const data = this.encoder.encode(message);
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        const hashArray = new Uint8Array(hashBuffer);
        // Быстрый перевод в hex
        let hex = '';
        for (let i = 0; i < hashArray.length; i++) {
            hex += hashArray[i].toString(16).padStart(2, '0');
        }
        return hex;
    }
}

new MiningWorker();