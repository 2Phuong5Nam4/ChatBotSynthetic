"""
Mock Tools cho Training Pipeline
Hash-based deterministic tools để đảm bảo cùng input cho cùng output
"""

import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class MockTools:
    """Mock implementation của tools được define trong extracted_procedure.json"""

    # Danh sách tên cửa hàng mẫu
    TEN_CUA_HANG = [
        "Tạp Hóa Bảo Trân", "Shop Chị Điệp", "Cửa Hàng Thanh Vinh",
        "TH Cô Hương", "Quán Hiếu Cáo", "Tạp Hóa Thượng Hải",
        "Cửa Hàng Nhã Vy", "Shop Tri Kỷ Quán", "Tạp Hóa Minh Khai",
        "CH Bình An", "Quán Phương Nam", "TH Hòa Bình"
    ]

    # Danh sách NPP/SubD mẫu
    NPP_SUBD = [
        ("10375694", "NPP QNI29"), ("10260229", "MH Phú Cường"),
        ("63401890", "SubD Cát Tiên"), ("10399522", "NPP CP8"),
        ("Đoàn Hằng", "SubD Đoàn Hằng"), ("Hiền Đô", "Supplier Hiền Đô"),
        ("Cường Hải", "Sub Cường Hải"), ("Cô Là", "Sub Cô Là")
    ]

    @staticmethod
    def _hash_input(input_str: str) -> int:
        """Hash string input thành integer để deterministic selection"""
        return int(hashlib.md5(input_str.encode()).hexdigest(), 16)

    @staticmethod
    def _select_from_list(seed: str, items: List, default=None):
        """Select item from list based on hash of seed"""
        if not items:
            return default
        hash_val = MockTools._hash_input(seed)
        return items[hash_val % len(items)]

    @staticmethod
    def _generate_sdt(seed: str) -> str:
        """Generate số điện thoại deterministic từ seed"""
        hash_val = MockTools._hash_input(seed)
        # Generate 10 digit phone starting with 0
        phone = "0" + str(hash_val % 900000000 + 100000000)[:9]
        return phone

    @staticmethod
    def _determine_status(outlet_id: str) -> str:
        """Determine status based on outlet_id hash"""
        hash_val = MockTools._hash_input(outlet_id)
        # 80% active, 15% đóng, 5% inactive
        prob = hash_val % 100
        if prob < 80:
            return "active"
        elif prob < 95:
            return "đóng"
        else:
            return "inactive"

    @staticmethod
    def tra_cuu_thong_tin(
        ma_cua_hang: Optional[str] = None,
        sdt: Optional[str] = None,
        ten_cua_hang: Optional[str] = None,
        ma_npp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tra cứu thông tin cửa hàng/NPP/SubD trên hệ thống

        Args:
            ma_cua_hang: Mã cửa hàng/OutletID 8 số
            sdt: Số điện thoại KH cung cấp để tra cứu (10 số)
            ten_cua_hang: Tên cửa hàng
            ma_npp: Mã NPP/SubD

        Returns:
            {
                "ma_cua_hang": str - Mã cửa hàng,
                "ten_cua_hang": str - Tên cửa hàng,
                "sdt_dang_ky": str - SĐT đăng ký HVN trong hệ thống (dùng để nhận OTP),
                "trang_thai": str - active/đóng/inactive,
                "da_dang_ky_app": bool - Đã đăng ký HVN chưa
            }

        Note:
            - SĐT input (sdt) là SĐT KH cung cấp để tra cứu
            - SĐT output (sdt_dang_ky) là SĐT đăng ký chính thức trong hệ thống
            - Trong hầu hết trường hợp, chúng giống nhau
            - Chỉ khác khi KH đã đổi số nhưng chưa cập nhật hệ thống
        """
        # Use first available input as seed
        seed = ma_cua_hang or sdt or ten_cua_hang or ma_npp or "default"

        # Generate deterministic ma_cua_hang if not provided
        if not ma_cua_hang:
            hash_val = MockTools._hash_input(seed)
            ma_cua_hang = str(60000000 + (hash_val % 10000000))  # 8-digit outlet ID

        # Select ten_cua_hang based on hash
        if not ten_cua_hang:
            ten_cua_hang = MockTools._select_from_list(
                ma_cua_hang,
                MockTools.TEN_CUA_HANG
            )

        # Generate SĐT
        sdt_dang_ky = MockTools._generate_sdt(ma_cua_hang)

        # Determine status
        trang_thai = MockTools._determine_status(ma_cua_hang)

        # 90% đã đăng ký app
        hash_val = MockTools._hash_input(ma_cua_hang + "app")
        da_dang_ky_app = (hash_val % 100) < 90

        return {
            "ma_cua_hang": ma_cua_hang,
            "ten_cua_hang": ten_cua_hang,
            "sdt_dang_ky": sdt_dang_ky,
            "trang_thai": trang_thai,
            "da_dang_ky_app": da_dang_ky_app
        }

    @staticmethod
    def kiem_tra_mqh(
        outlet_id: str,
        npp_subd_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Kiểm tra mối quan hệ (MQH) giữa Outlet với NPP/SubD trên SEM

        Returns:
            {
                "co_mqh": bool,
                "npp_subd_hien_tai": str,
                "ten_npp_subd": str,
                "trang_thai_mqh": str (Active/Inactive),
                "last_modified": str,
                "modified_by": str (SA/user/system),
                "tu_tao": bool
            }
        """
        seed = outlet_id + (npp_subd_id or "")
        hash_val = MockTools._hash_input(seed)

        # 70% có MQH
        co_mqh = (hash_val % 100) < 70

        if not co_mqh:
            return {
                "co_mqh": False,
                "npp_subd_hien_tai": None,
                "ten_npp_subd": None,
                "trang_thai_mqh": None,
                "last_modified": None,
                "modified_by": None,
                "tu_tao": False
            }

        # Select NPP/SubD
        npp_ma, npp_ten = MockTools._select_from_list(seed, MockTools.NPP_SUBD)

        # 85% Active, 15% Inactive
        trang_thai_mqh = "Active" if (hash_val % 100) < 85 else "Inactive"

        # 10% tự tạo
        tu_tao = (hash_val % 100) < 10

        # Modified by
        if tu_tao:
            modified_by = "user"
        else:
            modified_by = "SA" if (hash_val % 2) == 0 else "system"

        # Last modified (within last 48 hours)
        hours_ago = hash_val % 48
        last_modified_dt = datetime.now() - timedelta(hours=hours_ago)
        last_modified = last_modified_dt.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "co_mqh": True,
            "npp_subd_hien_tai": npp_ma,
            "ten_npp_subd": npp_ten,
            "trang_thai_mqh": trang_thai_mqh,
            "last_modified": last_modified,
            "modified_by": modified_by,
            "tu_tao": tu_tao
        }

    @staticmethod
    def kiem_tra_don_hang(
        ma_don_hang: str,
        kenh: str
    ) -> Dict[str, Any]:
        """
        Kiểm tra trạng thái đơn hàng trên hệ thống theo mã đơn và kênh đặt hàng.

        Sử dụng khi:
        - KH hỏi về trạng thái đơn đã đặt
        - Kiểm tra đơn không về NPP/SubD
        - Xác minh đơn Gratis đã được approve chưa
        - Tra cứu thông tin outlet và NPP/SubD liên quan đến đơn hàng

        Args:
            ma_don_hang: Mã đơn hàng cần kiểm tra
                Format thường gặp: số dài (VD: 2509076469100) hoặc mã có prefix (VD: CO251124-01481)
                Đây là thông tin bắt buộc phải hỏi KH trước khi gọi tool

            kenh: Kênh đặt hàng
                Giá trị hợp lệ: 'SEM' (Sales Force Execution Mobile),
                               'HVN' (Heineken Vietnam app),
                               'DIS_Lite' (Distributor Lite Portal)
                Phải hỏi KH để xác định kênh chính xác trước khi tra cứu

        Returns:
            {
                "trang_thai": str - Trạng thái hiện tại của đơn hàng
                    Possible values: "Đã nhận", "Đang xử lý", "Chưa về NPP", "Đã duyệt", "Chờ duyệt"
                    Usage: Sử dụng để phản hồi KH về tình trạng đơn
                           Nếu "Chưa về NPP" có thể cần kiểm tra MQH hoặc tạo ticket SEM

                "outlet_id": str - Mã cửa hàng (OutletID 8 số) đã đặt đơn này
                    Format: 8 chữ số (VD: 63235514)
                    Usage: Dùng để xác nhận đúng cửa hàng của KH, hoặc tra cứu thêm thông tin
                           qua tra_cuu_thong_tin() nếu cần

                "npp_subd": str - Mã NPP hoặc SubD được chỉ định nhận đơn hàng này
                    Format: 8 chữ số hoặc tên mã (VD: 10375694, Đoàn Hằng)
                    Usage: Xác định NPP/SubD nào phụ trách đơn
                           Có thể dùng để kiểm tra MQH qua kiem_tra_mqh() nếu có vấn đề

                "ngay_dat": str - Ngày đặt đơn hàng
                    Format: YYYY-MM-DD (VD: 2024-09-25)
                    Usage: Dùng để xác minh thời gian đặt đơn, tính toán thời gian xử lý,
                           hoặc kiểm tra quy tắc 24h đồng bộ

                "loai_don": str - Loại đơn hàng
                    Possible values: "Thường", "Gratis"
                    Usage: Phân biệt đơn thường và đơn khuyến mãi miễn phí
                           Nếu là "Gratis", kiểm tra thêm field "approved" và lưu ý
                           Gratis chỉ order qua NPP (không qua SubD)

                "approved": bool or None - Trạng thái approved của đơn Gratis
                    Chỉ có giá trị (true/false) khi loai_don="Gratis", còn lại là None
                    - true: Đơn Gratis đã được ASM approve
                    - false: Đơn Gratis chưa được approve hoặc bị từ chối
                    - None: Không áp dụng (đơn Thường)
                    Usage: Nếu loai_don="Gratis" và approved=false, hướng dẫn KH chờ ASM approve
                           Nếu approved=true, đơn đã sẵn sàng xử lý
            }

        Usage Notes:
            - Luôn hỏi KH về "mã đơn hàng" và "kênh" trước khi gọi tool này
            - Sau khi nhận kết quả, phản hồi KH về trạng thái đơn một cách rõ ràng
            - Nếu trang_thai="Chưa về NPP", cần kiểm tra MQH giữa outlet_id và npp_subd
              bằng kiem_tra_mqh()
            - Nếu loai_don="Gratis" và đơn order qua SubD, cảnh báo KH đây là lỗi
              (Gratis chỉ qua NPP)
            - Có thể dùng outlet_id từ kết quả để tra cứu thêm thông tin cửa hàng
              qua tra_cuu_thong_tin()

        Examples:
            # KH hỏi đơn không về NPP
            result = kiem_tra_don_hang(ma_don_hang="2509076469100", kenh="SEM")
            # → Xác nhận trang_thai → Nếu "Chưa về NPP" kiểm tra MQH → Tạo ticket nếu cần

            # Kiểm tra đơn Gratis đã approve chưa
            result = kiem_tra_don_hang(ma_don_hang="CO251124-01481", kenh="SEM")
            # → Kiểm tra loai_don="Gratis" → Xem approved=true/false → Phản hồi KH
        """
        seed = ma_don_hang + kenh
        hash_val = MockTools._hash_input(seed)

        # Generate outlet_id
        outlet_id = str(60000000 + (hash_val % 10000000))

        # Select NPP/SubD
        npp_ma, npp_ten = MockTools._select_from_list(seed, MockTools.NPP_SUBD)

        # Trạng thái đơn
        statuses = ["Đã nhận", "Đang xử lý", "Chưa về NPP", "Đã duyệt", "Chờ duyệt"]
        trang_thai = MockTools._select_from_list(seed + "status", statuses)

        # 20% Gratis, 80% Thường
        loai_don = "Gratis" if (hash_val % 100) < 20 else "Thường"

        # Nếu Gratis, 70% approved
        if loai_don == "Gratis":
            approved = (hash_val % 100) < 70
        else:
            approved = None

        # Ngày đặt (within last 7 days)
        days_ago = hash_val % 7
        ngay_dat_dt = datetime.now() - timedelta(days=days_ago)
        ngay_dat = ngay_dat_dt.strftime("%Y-%m-%d")

        return {
            "trang_thai": trang_thai,
            "outlet_id": outlet_id,
            "npp_subd": npp_ma,
            "ngay_dat": ngay_dat,
            "loai_don": loai_don,
            "approved": approved
        }

    @staticmethod
    def tao_ticket(
        team: str,
        noi_dung: str,
        du_lieu: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Tạo ticket chuyển tuyến cho team chuyên trách

        Args:
            team: Team xử lý (SEM/HVN/SA/CS/IT)
            noi_dung: Mô tả vấn đề
            du_lieu: Dữ liệu liên quan

        Returns:
            {
                "ticket_id": str,
                "trang_thai": str
            }
        """
        seed = team + noi_dung + json.dumps(du_lieu, sort_keys=True)
        hash_val = MockTools._hash_input(seed)

        # Generate ticket ID
        ticket_id = f"TKT{hash_val % 1000000:06d}"

        return {
            "ticket_id": ticket_id,
            "trang_thai": "Đã tạo thành công"
        }

    @staticmethod
    def force_sync(
        outlet_id: str,
        npp_subd_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Thực hiện Force Sync dữ liệu trên SEM

        Returns:
            {
                "thanh_cong": bool
            }
        """
        seed = outlet_id + (npp_subd_id or "")
        hash_val = MockTools._hash_input(seed)

        # 95% thành công
        thanh_cong = (hash_val % 100) < 95

        return {
            "thanh_cong": thanh_cong
        }

    @staticmethod
    def gui_huong_dan(
        loai_huong_dan: str
    ) -> Dict[str, Any]:
        """
        Gửi hướng dẫn/tài liệu SOP cho khách hàng

        Args:
            loai_huong_dan: xuất_gratis/đăng_nhập/quên_mật_khẩu/etc.

        Returns:
            {
                "da_gui": bool
            }
        """
        # Always succeed for mock
        return {
            "da_gui": True
        }


# Convenience functions for easy calling
def tra_cuu_thong_tin(**kwargs) -> Dict[str, Any]:
    """Wrapper function for tra_cuu_thong_tin"""
    return MockTools.tra_cuu_thong_tin(**kwargs)


def kiem_tra_mqh(**kwargs) -> Dict[str, Any]:
    """Wrapper function for kiem_tra_mqh"""
    return MockTools.kiem_tra_mqh(**kwargs)


def kiem_tra_don_hang(**kwargs) -> Dict[str, Any]:
    """Wrapper function for kiem_tra_don_hang"""
    return MockTools.kiem_tra_don_hang(**kwargs)


def tao_ticket(**kwargs) -> Dict[str, Any]:
    """Wrapper function for tao_ticket"""
    return MockTools.tao_ticket(**kwargs)


def force_sync(**kwargs) -> Dict[str, Any]:
    """Wrapper function for force_sync"""
    return MockTools.force_sync(**kwargs)


def gui_huong_dan(**kwargs) -> Dict[str, Any]:
    """Wrapper function for gui_huong_dan"""
    return MockTools.gui_huong_dan(**kwargs)


# Tool registry for dynamic calling
TOOL_REGISTRY = {
    "tra_cuu_thong_tin": tra_cuu_thong_tin,
    "kiem_tra_mqh": kiem_tra_mqh,
    "kiem_tra_don_hang": kiem_tra_don_hang,
    "tao_ticket": tao_ticket,
    "force_sync": force_sync,
    "gui_huong_dan": gui_huong_dan,
}


def call_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    Dynamically call a tool by name

    Args:
        tool_name: Name of the tool
        **kwargs: Tool parameters

    Returns:
        Tool output

    Raises:
        ValueError: If tool not found
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(TOOL_REGISTRY.keys())}")

    return TOOL_REGISTRY[tool_name](**kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Mock Tools - Deterministic Hash-based Output")
    print("=" * 60)

    # Test tra_cuu_thong_tin
    print("\n1. tra_cuu_thong_tin(ma_cua_hang='63235514')")
    result1 = tra_cuu_thong_tin(ma_cua_hang="63235514")
    print(json.dumps(result1, indent=2, ensure_ascii=False))

    # Test same input gives same output
    print("\n2. Same input - should give same output:")
    result1_again = tra_cuu_thong_tin(ma_cua_hang="63235514")
    print(f"Results match: {result1 == result1_again}")

    # Test kiem_tra_mqh
    print("\n3. kiem_tra_mqh(outlet_id='63235514', npp_subd_id='10375694')")
    result2 = kiem_tra_mqh(outlet_id="63235514", npp_subd_id="10375694")
    print(json.dumps(result2, indent=2, ensure_ascii=False))

    # Test kiem_tra_don_hang
    print("\n4. kiem_tra_don_hang(ma_don_hang='2509076469100', kenh='SEM')")
    result3 = kiem_tra_don_hang(ma_don_hang="2509076469100", kenh="SEM")
    print(json.dumps(result3, indent=2, ensure_ascii=False))

    # Test tao_ticket
    print("\n5. tao_ticket(team='SEM', noi_dung='Đơn không về NPP', du_lieu={...})")
    result4 = tao_ticket(
        team="SEM",
        noi_dung="Đơn không về NPP",
        du_lieu={"outlet_id": "63235514", "ma_don": "2509076469100"}
    )
    print(json.dumps(result4, indent=2, ensure_ascii=False))

    # Test force_sync
    print("\n6. force_sync(outlet_id='63235514', npp_subd_id='10375694')")
    result5 = force_sync(outlet_id="63235514", npp_subd_id="10375694")
    print(json.dumps(result5, indent=2, ensure_ascii=False))

    # Test gui_huong_dan
    print("\n7. gui_huong_dan(loai_huong_dan='xuất_gratis')")
    result6 = gui_huong_dan(loai_huong_dan="xuất_gratis")
    print(json.dumps(result6, indent=2, ensure_ascii=False))

    # Test dynamic calling
    print("\n8. Using call_tool() dynamic calling:")
    result7 = call_tool("tra_cuu_thong_tin", ma_cua_hang="67803609")
    print(json.dumps(result7, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("Testing Complete - All outputs are deterministic!")
    print("=" * 60)
