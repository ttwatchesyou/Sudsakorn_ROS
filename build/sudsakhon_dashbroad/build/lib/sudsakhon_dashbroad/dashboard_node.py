import rclpy
from rclpy.node import Node
from flask import Flask, render_template_string, jsonify, request
import subprocess
import threading
import os
import json

# --- Configuration ---
PORT = 5005
# ไฟล์ที่ใช้เก็บรายชื่อ Service (จะถูกสร้างในโฟลเดอร์ที่รัน)
STORAGE_FILE = os.path.expanduser('~/monitored_services.json')
DEFAULT_SERVICES = ["sudsakhon_tf.service"]

class DashboardNode(Node):
    def __init__(self):
        super().__init__('sudsakhon_dashboard_node')
        self.get_logger().info('Sudsakhon Dashboard Node has started with Persistent Storage!')
        
        # โหลดรายชื่อจากไฟล์ ถ้าไม่มีให้ใช้ค่าเริ่มต้น
        self.monitored_services = self.load_services_from_file()
        
        # เริ่ม Flask ใน Thread แยก
        self.flask_app = Flask(__name__)
        self.setup_routes()
        self.web_thread = threading.Thread(target=self.run_flask)
        self.web_thread.daemon = True
        self.web_thread.start()

    def load_services_from_file(self):
        """โหลดรายชื่อ Service จากไฟล์ JSON"""
        if os.path.exists(STORAGE_FILE):
            try:
                with open(STORAGE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.get_logger().error(f'Failed to load services file: {e}')
        return list(DEFAULT_SERVICES)

    def save_services_to_file(self):
        """บันทึกรายชื่อ Service ลงไฟล์ JSON"""
        try:
            with open(STORAGE_FILE, 'w') as f:
                json.dump(self.monitored_services, f)
        except Exception as e:
            self.get_logger().error(f'Failed to save services file: {e}')

    def run_flask(self):
        self.flask_app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)

    def setup_routes(self):
        @self.flask_app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.flask_app.route('/api/services', methods=['GET'])
        def get_services():
            data = []
            for svc in self.monitored_services:
                status = self.get_service_status(svc)
                data.append({"name": svc, "status": status})
            return jsonify(data)

        @self.flask_app.route('/api/control', methods=['POST'])
        def control_service():
            req = request.json
            action = req.get('action') 
            service_name = req.get('service')
            
            if service_name not in self.monitored_services:
                return jsonify({"status": "error", "message": "Service not found"}), 404
                
            cmd = ["sudo", "systemctl", action, service_name]
            try:
                subprocess.run(cmd, check=True)
                return jsonify({"status": "success"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.flask_app.route('/api/logs/<service_name>')
        def get_logs(service_name):
            cmd = ["journalctl", "-u", service_name, "-n", "50", "--no-pager"]
            try:
                result = subprocess.check_output(cmd).decode('utf-8')
                return jsonify({"logs": result})
            except:
                return jsonify({"logs": "Could not fetch logs."})

        @self.flask_app.route('/api/add_service', methods=['POST'])
        def add_service():
            new_svc = request.json.get('service')
            if new_svc and new_svc not in self.monitored_services:
                self.monitored_services.append(new_svc)
                self.save_services_to_file() # บันทึกลงไฟล์ทันที
                return jsonify({"status": "success"})
            return jsonify({"status": "error"}), 400

        @self.flask_app.route('/api/remove_service', methods=['POST'])
        def remove_service():
            svc_to_remove = request.json.get('service')
            if svc_to_remove in self.monitored_services:
                self.monitored_services.remove(svc_to_remove)
                self.save_services_to_file() # บันทึกลงไฟล์ทันที
                return jsonify({"status": "success"})
            return jsonify({"status": "error"}), 400

    def get_service_status(self, service_name):
        try:
            cmd = ["systemctl", "is-active", service_name]
            status = subprocess.check_output(cmd).decode('utf-8').strip()
            return status
        except:
            return "inactive"

# --- UI Template (เพิ่มปุ่มลบ Service) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sudsakhon Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f172a; color: white; }
        .card { background-color: #1e293b; border: 1px solid #334155; }
        .log-area { background-color: #000; font-family: monospace; font-size: 0.85rem; height: 450px; }
    </style>
</head>
<body class="p-6">
    <div class="max-w-5xl mx-auto">
        <header class="flex flex-col sm:flex-row justify-between items-center mb-8 gap-4">
            <div>
                <h1 class="text-3xl font-bold text-blue-400">Sudsakhon Dashboard</h1>
                <p class="text-slate-400 text-sm">Persistent Service Monitoring</p>
            </div>
            <div class="flex gap-2 w-full sm:w-auto">
                <input id="new-svc-input" type="text" placeholder="service_name.service" class="bg-slate-800 border border-slate-700 px-3 py-2 rounded flex-grow text-sm">
                <button onclick="addService()" class="bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded text-sm transition font-medium">Add</button>
            </div>
        </header>

        <div id="services-container" class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Cards will be here -->
        </div>
    </div>

    <!-- Modal for Logs -->
    <div id="log-modal" class="fixed inset-0 bg-black/80 hidden flex items-center justify-center p-4 z-50">
        <div class="bg-slate-900 w-full max-w-4xl max-h-[90vh] rounded-xl overflow-hidden flex flex-col shadow-2xl">
            <div class="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-800">
                <h2 id="log-title" class="text-xl font-semibold text-white">Service Logs</h2>
                <button onclick="closeLogs()" class="text-slate-400 hover:text-white p-2">✕ Close</button>
            </div>
            <pre id="log-content" class="p-4 overflow-y-auto flex-grow text-green-400 log-area whitespace-pre-wrap"></pre>
        </div>
    </div>

    <script>
        async function fetchServices() {
            try {
                const res = await fetch('/api/services');
                const services = await res.json();
                const container = document.getElementById('services-container');
                container.innerHTML = services.map(svc => `
                    <div class="card p-5 rounded-xl shadow-lg border-t-4 ${svc.status === 'active' ? 'border-t-emerald-500' : 'border-t-rose-500'}">
                        <div class="flex justify-between items-start mb-4">
                            <div class="overflow-hidden">
                                <h3 class="text-lg font-bold truncate pr-2 text-slate-100">${svc.name}</h3>
                                <div class="flex items-center gap-2 mt-1">
                                    <span class="w-2 h-2 rounded-full ${svc.status === 'active' ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}"></span>
                                    <span class="text-[10px] font-bold uppercase tracking-widest ${svc.status === 'active' ? 'text-emerald-400' : 'text-rose-400'}">
                                        ${svc.status}
                                    </span>
                                </div>
                            </div>
                            <button onclick="removeService('${svc.name}')" class="text-slate-500 hover:text-rose-400 transition" title="Remove from list">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                                  <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5 v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z"/>
                                  <path fill-rule="evenodd" d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z"/>
                                </svg>
                            </button>
                        </div>
                        <div class="flex gap-2 mb-3">
                            <button onclick="control('${svc.name}', 'start')" class="flex-1 text-[10px] bg-emerald-600 hover:bg-emerald-500 py-2 rounded font-bold transition">START</button>
                            <button onclick="control('${svc.name}', 'stop')" class="flex-1 text-[10px] bg-rose-600 hover:bg-rose-500 py-2 rounded font-bold transition">STOP</button>
                            <button onclick="control('${svc.name}', 'restart')" class="flex-1 text-[10px] bg-amber-600 hover:bg-amber-500 py-2 rounded font-bold transition">RESTART</button>
                        </div>
                        <button onclick="showLogs('${svc.name}')" class="w-full text-center py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm transition font-medium border border-slate-600">
                            View Logs
                        </button>
                    </div>
                `).join('');
            } catch (e) { console.error("Error fetching services", e); }
        }

        async function control(service, action) {
            try {
                await fetch('/api/control', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({service, action})
                });
                fetchServices();
            } catch (e) { console.error(e); }
        }

        async function removeService(service) {
            if(!confirm('Do you want to remove ' + service + ' from dashboard?')) return;
            try {
                await fetch('/api/remove_service', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({service})
                });
                fetchServices();
            } catch (e) { console.error(e); }
        }

        async function showLogs(service) {
            document.getElementById('log-title').innerText = 'Logs: ' + service;
            document.getElementById('log-content').innerText = 'Fetching latest logs...';
            document.getElementById('log-modal').classList.remove('hidden');
            try {
                const res = await fetch('/api/logs/' + service);
                const data = await res.json();
                document.getElementById('log-content').innerText = data.logs || "No logs found.";
                // Auto scroll to bottom
                const pre = document.getElementById('log-content');
                pre.scrollTop = pre.scrollHeight;
            } catch (e) {
                document.getElementById('log-content').innerText = "Error loading logs.";
            }
        }

        function closeLogs() {
            document.getElementById('log-modal').classList.add('hidden');
        }

        async function addService() {
            const input = document.getElementById('new-svc-input');
            const service = input.value.trim();
            if(!service) return;
            try {
                await fetch('/api/add_service', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({service})
                });
                input.value = '';
                fetchServices();
            } catch (e) { console.error(e); }
        }

        setInterval(fetchServices, 3000);
        fetchServices();
    </script>
</body>
</html>
"""

def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()