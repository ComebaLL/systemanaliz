
class ManagementRequest:
    """Входной поток: заявка на подготовку управленческого документа"""
    def __init__(self, department, doc_type, content, priority="обычный"):
        self.department = department          # отдел-инициатор
        self.doc_type = doc_type              # тип документа (приказ, отчёт, распоряжение)
        self.content = content                # содержание
        self.priority = priority
        self.is_valid = True


class ManagementDocument:
    """Объект документа (управленческий документ магазина)"""
    def __init__(self, request):
        self.id = id(self)
        self.department = request.department
        self.doc_type = request.doc_type
        self.content = request.content
        self.priority = request.priority
        self.status = "Черновик (проект)"
        self.is_approved = False
        self.is_registered = False


class DocumentManager:
    """Реализация блока А1: Формирование проекта управленческого документа"""
    def __init__(self, name):
        self.name = name

    def create_document_project(self, request: ManagementRequest) -> ManagementDocument:
        print(f"[Блок А1] Менеджер по документации {self.name} обрабатывает заявку "
              f"от отдела «{request.department}»...")
        print(f"          Тип документа: {request.doc_type}")
        print(f"          Содержание: {request.content}")

        # Имитация управления (регламент магазина)
        if not request.content or len(request.content) < 10:
            print("[Управление] Отклонено по регламенту: содержание заявки слишком краткое.")
            request.is_valid = False
            return None

        if request.priority == "срочный" and request.doc_type == "приказ":
            print("[Управление] Срочный приказ — повышенный контроль.")

        document = ManagementDocument(request)
        print(f"[Блок А1] Проект документа №{document.id} успешно сформирован. "
              f"Статус: {document.status}")
        return document


class Director:
    """Реализация блока А2: Проверка, утверждение и регистрация документа"""
    def __init__(self, name):
        self.name = name

    def verify_approve_and_register(self, document: ManagementDocument):
        if not document:
            print("[Блок А2] Ошибка: Документ не существует.")
            return

        print(f"[Блок А2] Директор {self.name} проверяет проект документа №{document.id}...")
        print(f"          Тип: {document.doc_type} | Отдел: {document.department}")

        # Имитация проверки по регламенту и законодательству
        document.is_approved = True
        document.status = "Утверждён директором"

        print(f"[Блок А2] Документ утверждён. Начинается регистрация...")
        document.is_registered = True
        document.status = "Зарегистрирован и введён в действие"

        print(f"[Выход системы] Процесс завершён.")
        print(f"                Статус документа: {document.status}")
        print(f"                Документ №{document.id} передан в архив и исполнителю.")


# --- Демонстрация работы системы ---
if __name__ == "__main__":
    print("=== Запуск симуляции процесса ===")
    print("Документальное обеспечение управленческой деятельности магазина электроники\n")

    # 1. Поступление Входа (Заявка от отдела)
    request = ManagementRequest(
        department="Отдел продаж",
        doc_type="Приказ",
        content="О проведении акции на смартфоны Samsung и Xiaomi с 10 по 20 октября 2026 года. "
                "Скидка 15%. Ответственный — старший менеджер Смирнов.",
        priority="срочный"
    )

    # 2. Инициализация Исполнителей (Механизмы)
    manager = DocumentManager(name="Елена Петрова")
    director = Director(name="Иванов А.С.")

    # 3. Выполнение блока А1
    project = manager.create_document_project(request)

    # 4. Выполнение блока А2
    if project:
        director.verify_approve_and_register(project)

    print("\n=== Процесс документального обеспечения завершён ===")